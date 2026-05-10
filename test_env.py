import sys
import torch

def main():
    print("=" * 40)
    print("        Environment Test Script")
    print("=" * 40)

    # Check Python version
    print(f"[+] Python version:")
    print(sys.version)
    print("-" * 40)

    # Check PyTorch version
    print(f"[+] PyTorch version: {torch.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"[+] CUDA available:  {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"[+] CUDA device count: {device_count}")

        for i in range(device_count):
            print(f"    - Device {i}: {torch.cuda.get_device_name(i)}")

        # Test basic GPU tensor operation
        try:
            print("-" * 40)
            print("[+] Testing GPU tensor calculation...")
            # Create tensors on GPU
            tensor_a = torch.rand(3, 3).cuda()
            tensor_b = torch.rand(3, 3).cuda()
            # Perform operation
            tensor_c = tensor_a @ tensor_b
            print("    [Success] Matrix multiplication on GPU working perfectly!")
            print("    Result tensor shape:", tensor_c.shape)
        except Exception as e:
            print(f"    [Error] GPU calculation failed: {e}")
    else:
        print("\n[!] WARNING: CUDA is not available. PyTorch is running on CPU.")

    print("=" * 40)

if __name__ == "__main__":
    main()

