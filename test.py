import torch
print(torch.cuda.is_available())  # Should print True if GPU is ready
print(torch.cuda.get_device_name(0))

