import torch
import torch.nn as nn
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        torch.ops.mylib.pipe_split.default()
        x = self.fc2(x)
        return x

x = torch.randn(1)
m = M()

ep = torch.export.export(m, args=(x,))
print(ep)


