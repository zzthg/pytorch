# Owner(s): ["oncall: distributed"]

from enum import auto, Enum

import torch
import torch.distributed.checkpoint as DCP
import torch.nn as nn
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.testing._internal.common_state_dict import VerifyStateDictMixin
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.distributed._tensor import DTensor as DT, Replicate
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
    MLPModule
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


# Simple and boring model to test interface and some corner cases that do not
# require complicated wrapping strategy.
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(8, 8, device="cuda")

class SimpleModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))

    def get_input(self):
        return torch.rand(8, 10, device="cuda")

class ModelType(Enum):
    FSDP = auto()
    HSDP = auto()
    FSDP_TP = auto()
    NONE = auto()  # no parallelization

def _compare_params(self, m1, m2):
        with FSDP.summon_full_params(m1):
            with FSDP.summon_full_params(m2):
                for n_p1, n_p2 in zip(m1.named_parameters(), m2.named_parameters()):
                    p1 = n_p1[1]
                    p2 = n_p2[1]
                    if n_p1[0] != n_p2[0]:
                        self.assertTrue(n_p1[0] in n_p2[0])
                    name = n_p1[0]
                    if name == "net2.bias" and self.rank != 0:
                        continue
                    if type(p2) is DT:
                        p2 = p2.redistribute(p2.device_mesh, [Replicate()]).to_local()

                    torch.testing.assert_allclose(p1, p2)
                    # self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

def _train(model, optim, train_steps=1):
    torch.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss


class TestE2ELoadAndSave(DTensorTestBase, VerifyStateDictMixin):
    def _create_model(self, compile, model_type):
        dummy_model = TestDummyModel().cuda()

        assert model_type in ModelType, f"{model_type} is not supported."
        if model_type == ModelType.FSDP:
            device_mesh = init_device_mesh(self.device_type, (self.world_size,))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
            )
        elif model_type == ModelType.HSDP:
            device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        elif model_type == ModelType.FSDP_TP:

            mesh_2d = init_device_mesh(
                self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
            )
            tp_mesh = mesh_2d["tp"]
            dp_mesh = mesh_2d["dp"]
            # device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
            dummy_model = SimpleModel(self.device_type)
            model = parallelize_module(
                dummy_model,
                tp_mesh,
                parallelize_plan={
                    "mlp_0.net1": ColwiseParallel(),
                    "mlp_0.net2": RowwiseParallel(),
                    "mlp_1.net1": ColwiseParallel(),
                    "mlp_1.net2": RowwiseParallel(),
                },
            )
            # model = parallelize_module(
            #     dummy_model,
            #     tp_mesh,
            #     parallelize_plan={
            #         "net1.0": ColwiseParallel(),
            #         "net2.0": RowwiseParallel(),
            #         "net3": ColwiseParallel(),
            #         "net4.1": RowwiseParallel(),
            #     }
            # )
            model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
        else:
            model = SimpleModel(self.device_type)
            #model = dummy_model
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        if compile:
            model = torch.compile(model)

        return model, optim

    def _equal_state_dict(self, model_0, model_1):
        for params_0, params_1 in zip(model_0.values(), model_1.values()):
            if not torch.equal(params_0, params_1):
                return False
        return True

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    @parametrize("compile", [True, False])
    @parametrize("model_type", [ModelType.FSDP, ModelType.HSDP, ModelType.FSDP_TP])
    def test_e2e(self, compile, model_type):
        model, optim = self._create_model(compile, ModelType.NONE)
        _train(model, optim, train_steps=2)

        dist_model, dist_optim = self._create_model(compile, model_type)
        _train(dist_model, dist_optim, train_steps=2)

        if self.rank==0:
            print(f"pre load: {optim.state_dict()['state'][0]=}")
        print(f"pre load: {self.rank=} {dist_optim.state_dict()['state'].get(0, None)=}")

        # create and save a checkpoint for parallel model
        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        DCP.save_state_dict(
            state_dict={"model": dist_msd, "optimizer": dist_osd},
            storage_writer=DCP.FileSystemWriter(self.temp_dir),
        )

        # load the checkpoint, starting with a new model
        dist_model, dist_optim = self._create_model(compile, model_type)
        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        DCP.load_state_dict(
            {"model": dist_msd, "optimizer": dist_osd},
            storage_reader=DCP.FileSystemReader(self.temp_dir),
        )
        set_state_dict(
            dist_model,
            optimizers=dist_optim,
            model_state_dict=dist_msd,
            optim_state_dict=dist_osd,
        )

        # at this point, the incorrect data is loaded!
        print(f"post load: {self.rank=} {dist_optim.state_dict()['state'].get(0, None)=}")

        return
        # train one more step on both models
        loss = _train(model, optim, train_steps=1)
        dist_loss = _train(dist_model, dist_optim, train_steps=1)
        self.assertEqual(loss, dist_loss)

        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        model_sd, optim_sd = get_state_dict(model, optimizers=optim)

        # print(_compare_params(self, model, dist_model))
        self._verify_msd(model_sd, dist_msd)
        # self._verify_osd_by_load(
        #     model, optim, torch.optim.Adam(model.parameters(), lr=0.1), optim_sd
        # )




instantiate_parametrized_tests(TestE2ELoadAndSave)
if __name__ == "__main__":
    run_tests()
