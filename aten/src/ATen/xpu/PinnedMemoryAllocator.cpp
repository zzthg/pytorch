#include <ATen/xpu/PinnedMemoryAllocator.h>
#include <ATen/Context.h>
#include <ATen/Config.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>

namespace at::native {

bool is_pinned_xpu(const Tensor& self, c10::optional<Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_cuda());
  // TODO: unhook this
  return detail::getXPUHooks().isPinnedPtr(self.storage().data());
}

} // namespace at::native
