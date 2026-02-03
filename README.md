# if you need to test function and class by using jupyter notebook or ipynb file in VS code

## please run

``` command line
uv run python -m ipykernel install --user --name llm-scratch --display-name "Python (llm-scratch)"
```

first then

``` command line
uv run jupyter lab
```

create the new notebook or **`\*.ipynb`** file

then select kernel

> Python (llm-scratch)

in vs code on the top right "Select Kernel" button

- go to "Jupyter kernel"
- click reload sign
- select "Python (llm-scratch)" kernel

## ปัญหาที่พบบ่อย

1. ไม่สามารถ import function ที่สร้างขึ้นเองใหม่ได้

    **Example**
    filename : **`main.py`**

    ```python filename="main.py"
    from llm_architect import DataProcessor,    helper_function, DATABASE_URL
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

    ```

    if you run **`uv sync`** command like

    ```command line
    $ uv sync
    Resolved 116 packages in 1ms
    Uninstalled 1 package in 5ms
     - llm-scratch==0.1.0 (from file:///D:/python/  llm-scratch)
    ```

    จะทำให้ระบบของ **`uv`** ดำเนินการจัดการ dependency   ตัวเองใหม่หมด ทำให้หา local dependency อย่าง   **`llm_architect`** ไม่เจอ

    ``` command line
    $ uv run main.py
    Traceback (most recent call last):
      File "D:\python\llm-scratch\main.py", line 1, in  <module>
        from llm_architect import DataProcessor,    helper_function, DATABASE_URL
    ModuleNotFoundError: No module named 'llm_architect'
    ```

    เราต้องแก้โดยการใช้คำสั่ง *uv pip install -e .*

    ```command line
    $ uv pip install -e .
    Resolved 113 packages in 488ms
          Built llm-scratch @ file:///D:/python/    llm-scratch
    Prepared 1 package in 3.27s
    ░░░░░░░░░░░░░░░░░░░░ [0/1] Installing   wheels...                                                          warning: Failed to hardlink files; falling     back to full copy. This may lead to degraded    performance. 
             If the cache and target directories are on     different filesystems, hardlinking may not  be supported.
             If this is intentional, set `export    UV_LINK_MODE=copy` or use `--link-mode=copy`   to suppress this warning.
    Installed 1 package in 38ms
     + llm-scratch==0.1.0 (from file:///D:/python/  llm-scratch)
     ```

    เพื่อบอกให้ uv เข้าไปหา local dependency ใหม่
