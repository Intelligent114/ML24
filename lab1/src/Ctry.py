import subprocess
import csv
import re
import os
import itertools
import shutil

# 定义学习率的取值范围
lr_values = [3e-9, 3.1e-9, 3.2e-9, 3.3e-9, 3.4e-9, 3.5e-9, 3.6e-9, 3.7e-9, 3.8e-9, 3.9e-9, 4e-9, 4.1e-9, 4.2e-9, 4.3e-9, 4.4e-9, 4.5e-9, 4.6e-9, 4.7e-9, 4.8e-9, 4.9e-9, 5e-9]  # 共 21 个值

# 固定的 mean 参数
mean_value = 4.201948

# 正确的虚拟环境 Python 解释器路径
venv_python = r'E:\dev\mech\1\USTC-ML24-Fall-main\.venv\Scripts\python.exe'

# 当前脚本所在目录
script_dir = r'E:\dev\mech\1\USTC-ML24-Fall-main\lab1\src'

# 结果文件的路径
results_csv = os.path.join(script_dir, 'resultsC.csv')

# 如果结果文件已存在，先删除
if os.path.exists(results_csv):
    os.remove(results_csv)

# 写入 CSV 文件的标题行
with open(results_csv, mode='w', newline='') as csv_file:
    fieldnames = ['lr', 'mean', 'Accuracy']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

# 依次遍历所有 lr 值
for lr in lr_values:
    print(f'正在处理 lr={lr}, mean={mean_value}')

    # 创建唯一的结果目录，添加 mean 值标记
    result_dir = os.path.join(script_dir, '..', 'results', 'train', f'lr_{lr}_mean_{mean_value}')
    os.makedirs(result_dir, exist_ok=True)

    # 训练命令
    train_command = [
        venv_python,
        os.path.join(script_dir, 'trainC.py'),
        '--results_path', result_dir,
        f'--mean={mean_value}',
        f'--lr={lr}'
    ]
    print('训练命令:', ' '.join(train_command))
    subprocess.run(train_command, cwd=script_dir)

    # **训练后，将结果文件复制到不带 '_Classification' 的目录中**
    train_result_dir = result_dir + '_Classification'
    if os.path.exists(train_result_dir):
        # 将训练结果文件复制到 result_dir
        for filename in os.listdir(train_result_dir):
            src_file = os.path.join(train_result_dir, filename)
            dst_file = os.path.join(result_dir, filename)
            shutil.copy2(src_file, dst_file)
    else:
        print(f'训练结果目录不存在：{train_result_dir}')

    # 评估命令，使用不带 '_Classification' 的目录
    eval_command = [
        venv_python,
        os.path.join(script_dir, 'evalC.py'),
        '--results_path', result_dir  # 使用不带 '_Classification' 的目录
    ]
    print('评估命令:', ' '.join(eval_command))
    result = subprocess.run(eval_command, cwd=script_dir, stdout=subprocess.PIPE, text=True)

    # 提取 Accuracy
    accuracy = None
    for line in result.stdout.split('\n'):
        match = re.search(r'Accuracy\s*[:=]\s*([0-9.]+)', line)  # 提取 Accuracy
        if match:
            accuracy = float(match.group(1))
            break

    # 将结果写入 CSV
    if accuracy is not None:
        with open(results_csv, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['lr', 'mean', 'Accuracy'])
            writer.writerow({
                'lr': lr,
                'mean': mean_value,
                'Accuracy': accuracy
            })
    else:
        print('未能提取 Accuracy，可能是评估输出格式有误。')
