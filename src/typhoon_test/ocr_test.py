from typhoon_ocr import ocr_document

def ocr_pdf_to_markdown(file_path: str) -> str:
    markdown = ocr_document(file_path, 
                            base_url="http://localhost:11434/v1", 
                            api_key="ollama",
                            model="scb10x/typhoon-ocr1.5-3b")
    
    return markdown