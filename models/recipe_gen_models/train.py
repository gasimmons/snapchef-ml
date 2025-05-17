import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.preprocessing import load_data, get_example


def main():
    df = load_data("/mnt/hdd/datasets/recipenlg/versions/1/dataset/full_dataset.csv")
    examples = get_example(df)
    print(examples)


if __name__ == "__main__":
    main()
