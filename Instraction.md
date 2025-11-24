#### Запустить распознавание
python PaddleOCR\tools\infer\predict_system.py --image_dir "F:\AllnAll\Project_3\OCR\Input\png_pre" --det_model_dir "F:\AllnAll\Project_3\OCR\PaddleOCR\inference\ch_PP-OCRv3_det_infer" --rec_model_dir "F:\AllnAll\Project_3\OCR\PaddleOCR\inference\cyrillic_PP-OCRv3_rec_infer" --rec_char_dict_path "F:\AllnAll\Project_3\OCR\PaddleOCR\ppocr\utils\dict\cyrillic_dict.txt" --rec_algorithm SVTR --rec_image_shape "3,48,320" --use_gpu false --vis_font_path "C:\Windows\Fonts\arial.ttf" --draw_img_save_dir "F:\AllnAll\Project_3\OCR\Output\ocr_vis_png"

Результат: распознанные картинки png в выходной папке + текстовый файл с результатами

#### Запустить пост-обработку:
f:/AllnAll/Project_3/OCR/postprocess/run_postprocess.py

Результат: файл по указанным путям с пост-обработанным текстом
Сам пайплайн пост-обработки указан в normalize_text.py

#### Сравнить метрики:
python F:\AllnAll\Project_3\OCR\postprocess\compare_ocr_metrics.py --ref "F:\AllnAll\Project_3\OCR\Output\ocr_vis_png\system_results.txt" --hyp "F:\AllnAll\Project_3\OCR\Output\ocr_vis_png\rec_results_post.txt"

Результат: вывод в консоль метрик сравнения

#### Запустить пост-обработку с помощью ЛЛМки:
f:/AllnAll/Project_3/OCR/postprocess/llm_correct_json.py




1) Накатить PaddleOCR
git clone https://github.com/PaddlePaddle/PaddleOCR.git
Или сами скачайте архив вот отсюда:
https://github.com/PaddlePaddle/PaddleOCR

2) Скачать то, что есть на гитхабе у нас
https://github.com/Mifikcha/Project_OCR

3) Поставить реквы
pip install -r requirements.txt  

4) Для ЛЛМКИ:
Я делал так: поставил gpt4all локально как приложение
https://docs.gpt4all.io/index.html
И потом через него скачал Qwen2-1.5B-Instract

Для ЛЛМКИ есть файл llm_correct_json.py
Он делает пост-обработку с помощью ЛЛМки, промт задается внутри скрипта
Результат: текстовый файл пост-обработки в выходной папке

Пообщаться с ЛЛМкой в консоли (или просто проверить что она ваще делает) можно с помощью скрипта llm_test.py



