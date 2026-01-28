# Follow this step

1. install uv

    ``` command line
    $ pip install uv
    for install uv
    ```

2. initial project

    ``` command line
    $ uv init project_name
    for initial project "project_name"
    ```

    ``` command line
    $ uv pip install -e .
    Resolved 113 packages in 2. 69s                                                                                                                                            
          Built llm-scratch @ file:///D:/python/llm-scratch
    Prepared 14 packages in 5.32s
    Uninstalled 1 package in    8ms                                                                                                                                                  
    ░░░░░░░░░░░░░░░░░░░░ [0/14] Installing  wheels...                                                                                                                          warning: Failed to hardlink files; falling back to  full copy. This may lead to degraded     performance.                                                                        
             If the cache and target directories are on     different filesystems, hardlinking may not be   supported.                                                               
             If this is intentional, set `export    UV_LINK_MODE=copy` or use `--link-mode=copy` to    suppress this  warning.                                                       
    Installed 14 packages in 1. 78s                                                                                                                                            
     + annotated-types==0.7.    0                                                                                                                                                  
     + distro==1.9. 0                                                                                                                                                             
     + ftfy==6.3.1
     + jiter==0.12.0
     ~ llm-scratch==0.1.0 (from file:///D:/python/llm-scratch)
     + openai==2.16.0
     + pillow==12.1.0
     + pydantic==2.12.5
     + pydantic-core==2.41.5
     + pypdf==6.6.2
     + sniffio==1.3.1
     + tqdm==4.67.1
     + typhoon-ocr==0.4.1
     + typing-inspection==0.4.2
    ```

3. create src folder under root directory
4. create library name **`llm_architect`** under **`/src`** folder
5. add file **`__init__.py`** in **`llm_architect`** directory
6. add another file like **`core.py`** in the same lavel as **`__init__.py`** file
7. in **`core.py`** file add the code such as

    ```python
    class DataProcessor:
    def __init__(self, name):
    self.name = name

    def helper_function(x):
    return x \* 2
    ```

8. add export function, variable or class in **`__init__.py`** file for make import cleaner

    ```python
    from .core import DataProcessor, helper_function
    ```

9. in the other file like **`main.py`** call function and class from library by using import like

    ```python
    from llm_architect import DataProcessor, helper_function, DATABASE_URL

    def main():
    print(f"Connecting to {DATABASE_URL}")
    processor = DataProcessor("Test")

        result = helper_function(10)
        print(result)
        print("Hello from llm-scratch!")

    if __name__ == "__main__":
    main()
    ```
