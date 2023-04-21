#Script converts scanned PDFs to plain text using OCR.
#Note: XHTML files need to be converted to PDF first - "save as PDF" produces better quality than "print to PDF".
#NOTE: Blurred PDFs and low contrast to background may cause serious problems in OCR

# Requirements:
# pip install pdf2image
# conda install -c conda-forge poppler #pdf manipulation required by pdf2image
# conda install -c conda-forge pytesseract
# conda install -c conda-forge tesseract
#Note: need to download language data from https://github.com/tesseract-ocr/tessdata and put it in path
##https://stackoverflow.com/questions/14800730/tesseract-running-error
##put the trained language data in a folder and then point tesseract to it by adding the following environment variable in Windows advanced system settings
##TESSDATA_PREFIX = "C:\Users\some path\tesseract"
#Documentation for Tesseract: https://tesseract-ocr.github.io/tessdoc/

import pytesseract #OCR engine
from pytesseract import Output
import pdf2image #converts
import glob #finds all paths to pdfs
import os

# global settings
language = 'pol'  # pol for Polish, en for English, etc.
pdfs = glob.glob(r"data\*.pdf")
output_path = "./output"

print(f'Beginning OCR script for language: {language}.')

for pdf_path in pdfs:
    print("Processing file " + str(f'{pdf_path[5:-4]}'))  # note the f-concatenation
    pages = pdf2image.convert_from_path(pdf_path, dpi=300, grayscale=True, use_pdftocairo=True)
    # pdftocairo is supposed to improve performance
    combinedtext: str = ""

    # Determine page orientation and adjust config accordingly
    # I did not check differences in output between the two formulas for normal, horizontal pages
    for pageNum,imgBlob in enumerate(pages):
        print("....page no. " + str(pageNum))
        if pytesseract.image_to_osd(imgBlob, output_type=Output.DICT)['orientation'] == 0:
            text = pytesseract.image_to_string(imgBlob, lang=language)
        else:
            # download osd.traineddata for this to work
            text = pytesseract.image_to_string(imgBlob, lang=language, config='--psm 1')
        # simple text cleaning
        cleantext = text.encode('utf-8', 'ignore').decode()  # removes strange characters via "ignore"
        cleantext = cleantext.replace("  ", " ")
        cleantext = cleantext.replace("  ", " ")
        cleantext = cleantext.replace("  ", " ")
        cleantext = cleantext.replace("\n ", "")
        cleantext = cleantext.replace("\n\n", "\n")
        cleantext = cleantext.replace("\n\n", "\n")
        cleantext = cleantext.replace("\n\n", "\n")
        cleantext = cleantext.replace("-\n", "")  # simple removal of word-breaks
        combinedtext = combinedtext + "\n" + cleantext

    with open(os.path.join(output_path, f'{pdf_path[5:-4]}.txt'), 'w', encoding='utf-8') as output_file:
        # Note: if pdf_path changes, substring needs to be adjusted.
       try:
           output_file.write(combinedtext)
           print("....Output file saved.")
       except:
           print("Error in writing file " + str(f'{pdf_path[5:-4]}.txt'))
           continue

