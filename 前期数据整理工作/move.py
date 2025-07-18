import os
import shutil

# 设置源文件夹和目标文件夹路径
source_path = r'D:\mypaper\multimodal\dataset\Liver\new'
ct_target_path = r'D:\mypaper\multimodal\dataset\Liver\new_no_cropped\CT'
dsa_target_path = r'D:\mypaper\multimodal\dataset\Liver\new_no_cropped\DSA'
txt_target_path = r'D:\mypaper\multimodal\dataset\Liver\new_no_cropped\TXT'

# 创建目标文件夹（如果不存在）
os.makedirs(ct_target_path, exist_ok=True)
os.makedirs(dsa_target_path, exist_ok=True)
os.makedirs(txt_target_path, exist_ok=True)

# 遍历父文件夹
for parent_folder in os.listdir(source_path):
    parent_path = os.path.join(source_path, parent_folder)
    
    if not os.path.isdir(parent_path):
        continue
    
    # 记录子文件夹名作为 id
    folder_id = parent_folder
    
    # 查找包含 "CT" 的孙文件夹
    for sub_folder in os.listdir(parent_path):
        sub_folder_path = os.path.join(parent_path, sub_folder)
        
        # if os.path.isdir(sub_folder_path) and "CT" in sub_folder:
        #     # 查找名为 "A" 的文件夹并复制到 CT 目标文件夹下，重命名为 id
        #     for ct_sub_folder in os.listdir(sub_folder_path):
        #         ct_sub_folder_path = os.path.join(sub_folder_path, ct_sub_folder)
                
        #         if os.path.isdir(ct_sub_folder_path) and ct_sub_folder == "A":
        #             target_ct_folder = os.path.join(ct_target_path, folder_id)
        #             # 复制文件夹 A 到目标目录并重命名
        #             shutil.copytree(ct_sub_folder_path, target_ct_folder)
        #             print(f"复制 CT 文件夹: {ct_sub_folder_path} 到 {target_ct_folder}")
        
        # 查找包含 "DSA" 的孙文件夹
        if os.path.isdir(sub_folder_path) and "DSA" in sub_folder:
            for grandchild_folder in os.listdir(sub_folder_path):
                grandchild_folder_path = os.path.join(sub_folder_path, grandchild_folder)
                
                if os.path.isdir(grandchild_folder_path) and "去骨" in grandchild_folder:
                    # 查找并复制 .jpg 和 .txt 文件
                    for file_name in os.listdir(grandchild_folder_path):
                        file_path = os.path.join(grandchild_folder_path, file_name)
                        
                        # 跳过 classes.txt 文件
                        if file_name == "classes.txt":
                            print(f"跳过文件: {file_path}")
                            continue
                        
                        # # 复制 .jpg 文件到 DSA 文件夹，并重命名为 id
                        # if file_name.endswith('.jpg'):
                        #     target_dsa_file = os.path.join(dsa_target_path, f"{folder_id}.jpg")
                        #     shutil.copy(file_path, target_dsa_file)
                        #     print(f"复制 .jpg 文件: {file_path} 到 {target_dsa_file}")
                        
                        # 复制 .txt 文件到 TXT 文件夹，并重命名为 id
                        if file_name.endswith('.txt'):
                            target_txt_file = os.path.join(txt_target_path, f"{folder_id}.txt")
                            shutil.copy(file_path, target_txt_file)
                            print(f"复制 .txt 文件: {file_path} 到 {target_txt_file}")
