import re
with open("data_grup.csv", "r", encoding="utf-8") as file:
    data = file.read()
cleaned_data = re.sub(r'[^A-Za-z0-9.,!?;:\s]', '', data)  # Hanya angka, huruf, dan tanda baca
with open("cleaned_data.csv", "w", encoding="utf-8") as file:
    file.write(cleaned_data)
