import subprocess
import csv
import re
import os

# lr 和 batch_size 的取值范围
lr_values = [3e-5, 4e-5]
batch_size_values = [2048]

# 虚拟环境的 Python 解释器路径
venv_python = r'E:\dev\mech\1\USTC-ML24-Fall-main\.venv\Scripts\python.exe'

# 当前脚本所在目录
script_dir = r'E:\dev\mech\1\USTC-ML24-Fall-main\lab1\src'

# 结果文件的路径
results_csv = os.path.join(script_dir, 'resultsR.csv')


# 如果结果文件已存在，先删除
if os.path.exists(results_csv):
    os.remove(results_csv)

# 写入 CSV 文件的标题行
with open(results_csv, mode='w', newline='') as csv_file:
    fieldnames = ['lr', 'batch_size', 'Mean Squared Error', 'Relative Error 1', 'Relative Error 2', 'R^2']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

# 存储各个统计信息
mse_stats = []
relative_error1_stats = []
relative_error2_stats = []
r2_stats = []

# 遍历 lr 和 batch_size 的所有组合
for lr_value in lr_values:
    for batch_size_value in batch_size_values:
        print(f'正在处理 lr={lr_value}, batch_size={batch_size_value}')

        # 训练结果的目录路径，保持原样
        train_result_dir = os.path.join(script_dir, '..', 'results', 'train', f'lr_{lr_value}_batch_{batch_size_value}')
        os.makedirs(train_result_dir, exist_ok=True)

        # 训练命令
        train_command = [
            venv_python,
            os.path.join(script_dir, 'trainR.py'),
            '--results_path', train_result_dir,
            f'--lr={lr_value}',
            f'--batch_size={batch_size_value}'
        ]
        print('训练命令:', ' '.join(train_command))
        subprocess.run(train_command, cwd=script_dir)

        # 评估命令，将路径加上 "_Regression"
        eval_result_dir = train_result_dir + "_Regression"
        os.makedirs(eval_result_dir, exist_ok=True)

        eval_command = [
            venv_python,
            os.path.join(script_dir, 'evalR.py'),
            '--results_path', eval_result_dir  # 使用带有 _Regression 后缀的路径
        ]
        print('评估命令:', ' '.join(eval_command))
        result = subprocess.run(eval_command, cwd=script_dir, stdout=subprocess.PIPE, text=True)

        # 提取 Mean Squared Error、Relative Error 1、Relative Error 2 和 R^2
        mean_squared_error = None
        relative_error1 = None
        relative_error2 = None
        r2 = None

        for line in result.stdout.split('\n'):
            mse_match = re.search(r'Mean Squared Error\s*\(MSE\)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            rel1_match = re.search(r'Relative Error 1\s*\(Mean difference\)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            rel2_match = re.search(r'Relative Error 2\s*\(Mean of abs differences\)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            r2_match = re.search(r'R\^2 Score\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)

            if mse_match:
                mean_squared_error = float(mse_match.group(1))
            if rel1_match:
                relative_error1 = float(rel1_match.group(1))
            if rel2_match:
                relative_error2 = float(rel2_match.group(1))
            if r2_match:
                r2 = float(r2_match.group(1))

        # 将结果写入 CSV
        if all(val is not None for val in [mean_squared_error, relative_error1, relative_error2, r2]):
            mse_stats.append(mean_squared_error)
            relative_error1_stats.append(relative_error1)
            relative_error2_stats.append(relative_error2)
            r2_stats.append(r2)

            with open(results_csv, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({
                    'lr': lr_value,
                    'batch_size': batch_size_value,
                    'Mean Squared Error': mean_squared_error,
                    'Relative Error 1': relative_error1,
                    'Relative Error 2': relative_error2,
                    'R^2': r2
                })
        else:
            print('未能提取结果，可能是评估输出格式有误。')
