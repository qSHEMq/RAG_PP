from paddleocr import PaddleOCR
ocr = PaddleOCR(device='cpu', lang='ru')
results = ocr.predict(r"D:\OCR_PP\RAG_PP\data\TORG-12\Screenshot_1.png")

for res in results:
    res.save_to_json("TORG-12.json")
    res.save_to_img("TORG-12_annotated/")
    print("✅ Готово!")
