import torch

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用 CPU")

# 创建一个张量并移动到 GPU
x = torch.randn(3, 3).to(device)
print(x)
