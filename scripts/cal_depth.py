import pysam
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse

description = "hi"

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-i", "--input", help="Relative or direct input directory path which stores input files(npy files) for CBSuiTe.", required=True)

required_args.add_argument("-o", "--output", help="Relative or direct output directory path to write CBSuiTe output file.", required=True)

opt_args.add_argument("-t", "--thread", help="Number of Thread. Recommended to set to the same number as CPUs to speed up processing. Default=4", default=4)
args = parser.parse_args()

input_path = "/public/ojsys/eye/sujianzhong/peiyf/cbsuite/demo_data/bam/"
# 指定BAM文件列表和bed文件
filenames = [file for file in os.listdir(input_path) if file.endswith('.bam')]

bed_file = "hg38_bin100.bed"
output_dir = "./demo_data/depth/"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取bed文件并存储区域
regions = []
with open(bed_file, 'r') as bed:
    for line in bed:
        chrom, start, end = line.strip().split()[:3]
        regions.append((chrom, int(start), int(end)))

def process_bam_file(filename):
    print(f"Processing {filename}")
    f = os.path.basename(filename)

    # 打开BAM文件
    bamfile = pysam.AlignmentFile(os.path.join(input_path, filename), "rb")

    # 索引BAM文件
    pysam.index(os.path.join(input_path, filename))

    # 打开输出文件
    with open(os.path.join(output_dir, f"{f}.txt"), 'w') as output_file:
        # 遍历每个区域并计算深度
        for chrom, start, end in regions:
            # 使用 count_coverage 方法计算该区域的覆盖度
            depth = bamfile.count_coverage(chrom, start, end, quality_threshold=0)
            # 计算该区域内的平均深度并写入文件
            avg_depth = sum(sum(depth[i]) for i in range(4)) / (end - start)
            output_file.write(f"{chrom}\t{start}\t{end}\t{avg_depth:.4f}\n")

    bamfile.close()

# 使用多线程处理BAM文件
with ThreadPoolExecutor(max_workers=args.thread) as executor:  # max_workers 设置为CPU线程数
    executor.map(process_bam_file, filenames)

print("Depth calculation completed.")
