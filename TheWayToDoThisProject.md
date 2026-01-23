**Follow this step**

1.  install uv

    > $ pip install uv

2.  initial project

    > $ uv init project_name

    > $ uv pip install -e .

3.  create src folder under root directory
4.  create library name "llm_architect" under /src folder
5.  add file "\_\_init\_\_.py" in "llm_architect" directory
6.  add another file like "core.py" in the same lavel as "\_\_init\_\_.py" file
7.  in "core.py" file add the code such as

    > class DataProcessor:
    > def \_\_init\_\_(self, name):
    > self.name = name
    >
    > def helper_function(x):
    > return x \* 2

8.  add export function, variable or class in "\_\_init\_\_.py" file for make import cleaner

    > from .core import DataProcessor, helper_function

9.  in the other file like "main.py" call function and class from library by using import like

    > from llm_architect import DataProcessor, helper_function, DATABASE_URL
    >
    > def main():
    > print(f"Connecting to {DATABASE_URL}")
    > processor = DataProcessor("Test")
    >
    >     result = helper_function(10)
    >     print(result)
    >     print("Hello from llm-scratch!")
    >
    > if \_\_name\_\_ == "\_\_main\_\_":
    > main()
