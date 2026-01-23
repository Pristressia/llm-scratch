from llm_architect import DataProcessor, helper_function, DATABASE_URL
from llm_operator import test

def main():
    print(f"Connecting to {DATABASE_URL}")
    processor = DataProcessor("Test")

    result = helper_function(10)
    print(result)
    print("Hello from llm-scratch!")

    print(test(3000))
    


if __name__ == "__main__":
    main()
