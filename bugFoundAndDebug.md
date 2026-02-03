# Bug found and the way to debug in this project

1. **Circular module called**
    เกิดเมื่อมีการคอลฟังก์ชันหรือคลาสภายใน module เดียวกัน อาทิเช่น
    **`src/transformer/attention/Attention.py`**

    ```python
    import numpy as np
    import numpy.typing as npt
    from dataclasses import dataclass 
    from llm_operator import softmax, Softmax_cache,    layer_norm, Layer_norm_cache, softmaxBackward,     layer_norm_backward
    from transformer.transformerCore.TransformerCore    import TransformerCore
    ```

    **`src/transformer/transformerCore/transformerCore.py`**

    ```python
    import numpy as np
    import numpy.typing as npt

    class TransformerCore:
        pass

    ```

    จะเห็นได้ว่า **`attention.py`** มีการ import class **`TransformerCore`** ซึ่งอยู่ภายใต้ module เดียวกัน (*transformer*) หากใช้การ import สั้น ๆ อาทิ

    **`attention.py`**

    ```python
    from transformer import TransformerCore
    ```

    จะทำให้เกิดบัค ***circular import*** ขึ้นเมื่อเรา import class **Attention** จากการพยายาม import class ภายใน ของ python อธิบายอย่างง่ายคือ

    > called_file.py>transformer>attention>transformer>...
