import zipfile
import os
import pandas as pd

zip_path='D:\Desktop\cryptocurrencypricehistorydataset.zip'
extract_folder='D:\Desktop\ML Model'

with zipfile.ZipFile(zip_path,'r') as zip_ref:
    zip_ref.extractall(extract_folder)

csv_files=[os.path.join(extract_folder,file) for file in os.listdir(extract_folder) if file.endswith('.csv')]
print("Total Cryptocurrencies:" ,len(csv_files))