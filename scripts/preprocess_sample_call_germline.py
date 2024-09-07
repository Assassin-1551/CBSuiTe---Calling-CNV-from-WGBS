import numpy as np
import os
from os import listdir
import pandas as pd
from tqdm import tqdm
import argparse

from utils import message, chr_encode, label_encode_str, filter_lowqua_data, calculate_mean_std


description = "hi"

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-rd", "--readdepth", help="Please provide the readdepth path.", required=True)
required_args.add_argument("-o", "--output", help="Please provide the output path for the preprocessing.", required=True)
required_args.add_argument("-methy", "--methy", help="Please provide the methylation path.", required=True)
opt_args.add_argument("-down", "--downsample", help="Downsample Test.")

parser.add_argument("-V", "--version",   help="show program version", action="store_true")
args = parser.parse_args()

readdepths_path = args.readdepth
os.makedirs(args.output, exist_ok=True)
output_path = args.output

cur_dirname = os.path.dirname(__file__)

if args.downsample:
    down = args.downsample
    readdepths_file = [file for file in listdir(readdepths_path) 
                                 if not file.startswith('.') and file.endswith(f'.{down}.bam.txt')]
else:
    readdepths_file = [file for file in listdir(readdepths_path) if not file.startswith('.')]
#message(f"readdepths_file: {readdepths_file}")
readdepths_file.sort()
methy_file = [file for file in listdir(args.methy) if not file.startswith('.') and file.endswith('.bed')]
methy_file.sort()

binsize = 100000
depthsize = 100
seq_length = int(binsize / depthsize)

# 注意！！target_data，depth_list，label_list的染色体顺序必须一致！
chromosome_lengths = {
    1: 248956422, 10: 133797422, 11: 135086622, 12: 133275309, 13: 114364328, 
    14: 107043718, 15: 101991189, 16: 90338345, 17: 83257441, 18: 80373285, 
    19: 58617616, 2: 242193529, 20: 64444167, 21: 46709983, 22: 50818468, 
    3: 198295559, 4: 190214555, 5: 181538259, 6: 170805979, 7: 159345973, 
    8: 145138636, 9: 138394717, #23: 156040895, 24: 57227415,  #X：23， Y：24
}

# 创建target bed
num_rows = sum(length // binsize + 1 for length in chromosome_lengths.values())
target_data = np.zeros((num_rows, 3), dtype=int)
index = 0
for chromosome, length in chromosome_lengths.items():
    for start in (range(2781469, length, binsize) if chromosome == 24 else range(0, length, binsize)):
        end = length if start + binsize > length else start + binsize
        target_data[index] = [chromosome, start, end]
        index += 1
target_data = target_data[~np.all(target_data == 0, axis=1)]
print('target_data: ',np.shape(target_data))

gc_data = pd.read_csv('./demo_data/gc/GC.bed', sep="\t")
gc_values = gc_data.iloc[:, 4]
gc_values = np.array(gc_values).reshape(-1, 1)
print('gc_values: ',np.shape(gc_values))
    
for i, file_name in enumerate(readdepths_file):
    sample_name = file_name.split(".txt")[0]
    if "_" in sample_name:
        sample_name = sample_name.split("_")[0]

    name_list = [sample_name] * len(target_data)
    print('name_list: ',np.shape(name_list))
    
    methy_path = methy_file[i]

    for i in range(1, 23):
        array_name = f'methy_{i}'
        array_length = chromosome_lengths[i] // depthsize + 1
        #print(f"chr{i} length: {array_length}")
        locals()[array_name] = np.zeros(array_length, dtype=np.float)
    methy_dict = {
        1: methy_1, 10: methy_10, 11: methy_11, 12: methy_12, 13: methy_13,
        14: methy_14, 15: methy_15, 16: methy_16, 17: methy_17, 18: methy_18,
        19: methy_19, 2: methy_2, 20: methy_20, 21: methy_21, 22: methy_22,
        3: methy_3, 4: methy_4, 5: methy_5, 6: methy_6, 7: methy_7,
        8: methy_8, 9: methy_9, #23: methy_23, 24: methy_24  # X: 23, Y: 24
    }
    # 读取read depth 文件，并写入对应染色体的对应位置
    message(f"methy_files: {methy_path}")
    with open(os.path.join(args.methy, methy_path), 'r') as f:
        header = next(f)
        for line in f:
            value = line.strip().split('\t')
            chr, pos = chr_encode(value[0]), int(int(value[1])//depthsize)
            #print(chr, pos)
            met = 0 if value[3] == 'NA' else float(value[3])
            if int(chr) > 22:
                break
            methy_cur = methy_dict.get(int(chr))
            methy_cur[pos] = met
    # 划分区域
    methy_list = []
    for key, data in methy_dict.items():
        for i in range(0, len(data), seq_length):
            methy_arr = data[i : i+seq_length]
            methy_list.append(methy_arr)
    print('methy_list: ', np.shape(methy_list))


    # 创建24个数组用于depth匹配:depth_1, depth_2, ..., depth_24
    for i in range(1, 23):
        array_name = f'depth_{i}'
        array_length = chromosome_lengths[i] // depthsize + 1
        #print(f"chr{i} length: {array_length}")
        locals()[array_name] = np.zeros(array_length, dtype=np.float)
    depth_dict = {
        1: depth_1, 10: depth_10, 11: depth_11, 12: depth_12, 13: depth_13,
        14: depth_14, 15: depth_15, 16: depth_16, 17: depth_17, 18: depth_18,
        19: depth_19, 2: depth_2, 20: depth_20, 21: depth_21, 22: depth_22,
        3: depth_3, 4: depth_4, 5: depth_5, 6: depth_6, 7: depth_7,
        8: depth_8, 9: depth_9, #23: depth_23, 24: depth_24  # X: 23, Y: 24
    }

    # 读取read depth 文件，并写入对应染色体的对应位置
    file_path = os.path.join(readdepths_path, file_name)
    message(f"readdepth_files: {file_name}")
    with open(file_path, 'r') as f:
        header = next(f)
        for line in f:
            value = line.strip().split('\t')[:4]
            chr, pos, dep = chr_encode(value[0]), int(int(value[1])//depthsize), float(value[3])
            if int(chr) > 22:
                break
            depth_cur = depth_dict.get(int(chr))
            depth_cur[pos] = dep
    # 划分区域
    depth_list = []
    for key, data in depth_dict.items():
        for i in range(0, len(data), seq_length):
            depth_arr = data[i : i+seq_length]
            #depth_arr = filter_lowqua_data(depth_arr)
            depth_list.append(depth_arr)

    flattened_depth = np.concatenate(depth_list)
    non_zero_elements = flattened_depth[flattened_depth != 0]
    total_mean = np.mean(non_zero_elements)
    total_std = np.std(non_zero_elements)
    print("mean: ", total_mean, "std: ", total_std)

    z_normalized_list = []
    count = 0 
    for elem in depth_list:
    # Check if sum of element is not 0
        if np.sum(elem) != 0:
            z_normalized_elem = (elem - total_mean) / total_std     # Z-score normalize
            z_normalized_list.append(z_normalized_elem)
        else:
            # If sum of element is 0, append the element as it is
            z_normalized_list.append(elem)
            count += 1
    print('low qua array: ', count)
    print('depth_list: ', np.shape(depth_list))
    #print('depth_list: ', depth_list[0])
    
    name_list = np.array(name_list).reshape(-1, 1)
    z_normalized_list = np.array(z_normalized_list).reshape(-1, 1)
    methy_list = np.array(methy_list).reshape(-1, 1)
    assert name_list.shape[0] == target_data.shape[0] == z_normalized_list.shape[0]
    final_data = np.hstack((name_list, target_data, z_normalized_list, gc_values, methy_list,))
    print('final_data: ',np.shape(final_data))

    # 筛选掉depth为全0的行
    filtered_data = []

    for row in final_data:
        sum_of_array = np.sum(row[4])
        #if sum_of_array != 0 and row[0]!= 23 and row[0]!= 24:
        if sum_of_array != 0:
            filtered_data.append(row)

    # 将筛选后的数据转换回NumPy数组
    final_data = np.array(filtered_data)
    print('final_data: ',np.shape(final_data))

    if args.downsample:
        down = args.downsample
        np.save(os.path.join(output_path, f"{sample_name}_{down}_final_data.npy"), final_data)
    else:
        np.save(os.path.join(output_path, f"{sample_name}_final_data.npy"), final_data)
    print(f'{sample_name} preprocess complete!')
    print("____________________")


mean, std = calculate_mean_std(output_path)

with open("germline_stats.txt", 'w') as f:
    f.write(f"{mean},{std}")
    f.close()
