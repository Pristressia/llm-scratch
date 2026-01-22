from llm_architect import DataProcessor, helper_function, DATABASE_URL

def main():
    print(f"Connecting to {DATABASE_URL}")
    processor = DataProcessor("Test")

    result = helper_function(10)
    print(result)
    print("Hello from llm-scratch!")


if __name__ == "__main__":
    main()
