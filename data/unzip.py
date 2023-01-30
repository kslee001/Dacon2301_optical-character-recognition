import zipfile
with zipfile.ZipFile('open.zip', 'r') as z:
    z.extractall("/home/gyuseonglee/workspace/2301_OCR/data")