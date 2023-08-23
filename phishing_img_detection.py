from tika import unpack
import pytesseract as pt
import numpy as np

import cv2
import re

BLACLIST_WORDS = [
    'macros', 
    'mocros',
    'mocras',
    'macras'
]

def text_purification(text: str) -> str:
    purified_text = re.sub(r'[^a-zA-Z]', ' ', text)
    purified_text = purified_text.lower()
    purified_text = ' '.join(purified_text.split())
    
    return purified_text

def is_bl_word_detected(text: str) -> bool:
    for word in text.split():
        if word in BLACLIST_WORDS:
            return True
    return False

def image_processing(image: cv2.Mat) -> cv2.Mat:
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)

    return cv2.dilate(binary, kernel, iterations=1)


if __name__ == "__main__":
    file_path = input("File path: ")
    for _, attachment in unpack.from_file(file_path)['attachments'].items():
        image = cv2.imdecode(np.frombuffer(attachment, np.uint8), cv2.IMREAD_UNCHANGED)

        text = pt.image_to_string(image_processing(image))
        purified = text_purification(text)
        
        if is_bl_word_detected(purified):
            print("Phishing image was detected\nThis image was saved in the curent directory")
            print(text)
            cv2.imwrite("phishing_img.png", image)
        else:
            print("Nothing was found")
