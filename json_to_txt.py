import json

# 读取 JSON 文件
with open(r"C:/Users/Yorushika/Desktop/399picture(1)/399 picture/json/1.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 将 JSON 数据写入 TXT 文件
with open(r"C:/Users/Yorushika/Desktop/399picture(1)/399 picture/json/1.txt", 'w', encoding='utf-8') as txt_file:
    txt_file.write(json.dumps(data, indent=4, ensure_ascii=False))
