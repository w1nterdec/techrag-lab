import torch
import transformers
import datasets
import peft


def main():
    print("=== TechRAG-Lab Environment Check ===")
    print(f"PyTorch:       {torch.__version__}")
    print(f"Transformers:  {transformers.__version__}")
    print(f"Datasets:      {datasets.__version__}")
    print(f"PEFT:          {peft.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
