import subprocess
import csv
import re
import os
import itertools
import shutil

# 固定的 lr 和 batch_size 参数
lr_value = 6.95e-5
batch_size_value = 1024

# landepen 参数的取值范围
landepen_values = list(range(-1, -21, -1))  # 从 -1 到 -20，步长为 1

# 虚拟环境的 Python 解释器路径
venv_python = r'E:\dev\mech\1\USTC-ML24-Fall-main\.venv\Scripts\python.exe'

# 当前脚本所在目录
script_dir = r'E:\dev\mech\1\USTC-ML24-Fall-main\lab1\src'

# 结果文件的路径
results_csv = os.path.join(script_dir, 'resultslandepen.csv')

# 如果结果文件已存在，先删除
if os.path.exists(results_csv):
    os.remove(results_csv)

# 写入 CSV 文件的标题行
with open(results_csv, mode='w', newline='') as csv_file:
    fieldnames = ['lr', 'batch_size', 'landepen', 'Relative Error']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

# 依次遍历所有 landepen 值
for landepen in landepen_values:
    print(f'正在处理 lr={lr_value}, batch_size={batch_size_value}, landepen={landepen}')

    # 创建唯一的结果目录，包含 lr、batch_size 和 landepen 值
    result_dir = os.path.join(script_dir, '..', 'results', 'train', f'lr_{lr_value}_batch_{batch_size_value}_landepen_{landepen}')
    os.makedirs(result_dir, exist_ok=True)

    # 训练命令
    train_command = [
        venv_python,
        os.path.join(script_dir, 'trainR.py'),
        '--results_path', result_dir,
        f'--lr={lr_value}',
        f'--batch_size={batch_size_value}',
        f'--landepen={landepen}'
    ]
    print('训练命令:', ' '.join(train_command))
    subprocess.run(train_command, cwd=script_dir)

    # **训练后，将结果文件复制到不带 '_Regression' 的目录中**
    train_result_dir = result_dir + '_Regression'
    if os.path.exists(train_result_dir):
        # 将训练结果文件复制到 result_dir
        for filename in os.listdir(train_result_dir):
            src_file = os.path.join(train_result_dir, filename)
            dst_file = os.path.join(result_dir, filename)
            shutil.copy2(src_file, dst_file)
    else:
        print(f'训练结果目录不存在：{train_result_dir}')

    # 评估命令，使用不带 '_Regression' 的目录
    eval_command = [
        venv_python,
        os.path.join(script_dir, 'evalR.py'),
        '--results_path', result_dir  # 使用不带 '_Regression' 的目录
    ]
    print('评估命令:', ' '.join(eval_command))
    result = subprocess.run(eval_command, cwd=script_dir, stdout=subprocess.PIPE, text=True)

    # 提取 Relative Error
    relative_error = None
    for line in result.stdout.split('\n'):
        match = re.search(r'Relative Error\s*[:=]\s*([0-9.]+)', line)  # 提取 Relative Error
        if match:
            relative_error = float(match.group(1))
            break

    # 将结果写入 CSV
    if relative_error is not None:
        with open(results_csv, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['lr', 'batch_size', 'landepen', 'Relative Error'])
            writer.writerow({
                'lr': lr_value,
                'batch_size': batch_size_value,
                'landepen': landepen,
                'Relative Error': relative_error
            })
    else:
        print('未能提取 Relative Error，可能是评估输出格式有误。')
