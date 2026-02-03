from llm_architect import DataProcessor, helper_function, DATABASE_URL
from llm_operator import test
from transformer import Attention, Attention_init
import numpy as np

def main():
    print(f"Connecting to {DATABASE_URL}")
    processor = DataProcessor("Test")

    result = helper_function(10)
    print(result)
    print("Hello from llm-scratch!")

    print(test(3000))

    rng = np.random.default_rng(1000)

    B, T, C = 2, 4, 6

    attention_init = Attention_init.randomInitial(1000, B, T, C)
    attention1 = Attention(attention_init, "attention1")

    X = rng.random((B, T, C), dtype=np.float64)
    print(attention1.forward_train(X))


    


if __name__ == "__main__":
    main()
