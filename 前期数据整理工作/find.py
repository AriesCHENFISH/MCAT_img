import os
import pandas as pd
import fnmatch
from tqdm import tqdm  # 导入 tqdm 进度条模块

# 读取Excel文件中的“住院号”列
file_path = r"D:\论文\多模态工作\dataset\Liver\TACE治疗反应评价.xlsx"
df = pd.read_excel(file_path)

# 确保“住院号”列存在
if "住院号" not in df.columns:
    print("Excel 文件中没有找到 '住院号' 这一列")
else:
    # 初始化结果列表
    results = []

    # 提示开始处理
    print("开始处理住院号数据...")

    # 创建进度条，遍历“住院号”列
    for inpatient_number in tqdm(df["住院号"], desc="正在查找匹配的文件夹", ncols=100):
        # 将住院号转为字符串并确保没有额外空白，取前10个字符
        inpatient_number = str(inpatient_number).strip()[:10]

        # 标记是否找到匹配文件夹
        found_folders = []

        # 遍历指定目录（这里是Liver目录）所有文件夹
        for root, dirs, files in os.walk("D:/论文/多模态工作/dataset/Liver"):  # 遍历整个Liver目录
            for folder in dirs:
                # 使用 fnmatch 进行模糊匹配，匹配文件夹名以住院号开头
                if fnmatch.fnmatch(folder, f"{inpatient_number}*"):
                    # 找到匹配的文件夹，记录其路径
                    folder_path = os.path.join(root, folder)
                    found_folders.append(folder_path)

        # 如果找到了匹配文件夹，则记录这些文件夹路径，否则记录“文件夹不存在”
        if found_folders:
            # 将住院号和所有路径保存到一行中，路径命名为路径1、路径2等
            row = [inpatient_number]
            for i, folder_path in enumerate(found_folders, start=1):
                row.append(folder_path)
            results.append(row)
        else:
            # 如果没有找到匹配文件夹，保存住院号和"文件夹不存在"
            results.append([inpatient_number, "文件夹不存在"])

    # 处理完成后，提示
    print("\n查找完成，正在保存结果到 Excel 文件...")

    # 将结果保存到新的 Excel 文件中
    # 动态调整列名
    max_paths = max(len(row) - 1 for row in results)  # 计算最大路径列数
    columns = ["住院号"] + [f"路径{i}" for i in range(1, max_paths + 1)]

    results_df = pd.DataFrame(results, columns=columns)
    output_path = r"D:\论文\多模态工作\dataset\Liver\匹配结果.xlsx"
    results_df.to_excel(output_path, index=False)

    # 提示保存成功
    print(f"匹配结果已保存到：{output_path}")
