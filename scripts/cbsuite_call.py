'''
CBSuiTe source code.
Training
'''
import numpy as np
from tensorflow.keras.preprocessing import sequence
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import pandas as pd
import os
from tqdm import tqdm
import argparse
import gc

from net_my_gm_v2 import CNVcaller
from utils import message


cur_dirname = os.path.dirname(__file__)

''' 
Perform I/O operations.
'''

description = "hi"

parser = argparse.ArgumentParser(description=description)

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-m", "--model", help="If you want to use pretrained CBSuiTe weights choose one of the options: \n \
                   (i) germline \n (ii) somatic. \n Or you can use your own trained model by giving path/to/your/model.pt ", required=True)

required_args.add_argument("-i", "--input", help="Relative or direct input directory path which stores input files(npy files) for CBSuiTe.", required=True)

required_args.add_argument("-o", "--output", help="Relative or direct output directory path to write CBSuiTe output file.", required=True)

required_args.add_argument("-n", "--normalize", help="Please provide the path for mean&std stats of read depth values to normalize. \n \
                                                    These values are obtained precalculated from the training dataset.", required=True)

opt_args.add_argument("-g", "--gpu", help="Specify gpu", required=False)
opt_args.add_argument("-bs", "--batch_size", help="Batch size used in calling.",default=16, required=False)
opt_args.add_argument("-V", "--version", help="show program version", action="store_true")

args = parser.parse_args()

if args.version:
    print("CBSuiTe version 0.1")

if args.gpu:
    message("Trying to use GPU!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    message("Using CPU!")
    device = torch.device("cpu")
print(f"Using device: {device}")

os.makedirs(args.output, exist_ok=True)
print("Output Dir: ",args.output)

bs = int(args.batch_size)

model = CNVcaller(1000, 1, 3, 192, 3)

if args.model == "germline":
    model.load_state_dict(torch.load(os.path.join(cur_dirname, "../model/germline/CBSuiTe_germline.pt"), map_location = device))
elif args.model == "somatic":
    model.load_state_dict(torch.load(os.path.join(cur_dirname, "../model/somatic/CBSuiTe_somatic.pt"), map_location = device))
else:
    model.load_state_dict(torch.load(args.model, map_location = device))

print("load model: ", args.model)


model.eval()
model = model.to(device)

input_files = os.listdir(args.input)
all_samples_names = [file.split("_final_data.npy")[0] for file in input_files]

message("Calling for CNV regions...")

for sample_name in tqdm(all_samples_names):
    
    message(f"calling sample: {sample_name}")

    sampledata = np.load(os.path.join(args.input, sample_name+"_final_data.npy"), allow_pickle=True)
    
    sampnames_data = []
    chrs_data = []
    readdepths_data = []
    start_inds_data = []
    end_inds_data = []

    temp_sampnames = sampledata[:,0]
    temp_chrs = sampledata[:,1]
    temp_start_inds = sampledata[:,2]
    temp_end_inds = sampledata[:,3]
    temp_readdepths = sampledata[:,4]
    temp_gc = sampledata[:,5]
    temp_methy = sampledata[:,6]


    file1 = open(args.normalize, 'r')
    line = file1.readline()
    means_ = float(line.split(",")[0])
    stds_ = float(line.split(",")[1])

    print(means_, stds_)

    for i in range(len(temp_chrs)):
        
        temp_readdepths[i] = list(temp_readdepths[i])
        temp_methy[i] = list(temp_methy[i])
        combined_list = temp_readdepths[i] + temp_methy[i]
        combined_list.extend([temp_gc[i], temp_end_inds[i], temp_start_inds[i], temp_chrs[i] ])   #在序列末尾插入个0，end，start和chr
        temp_readdepths[i] = combined_list

    sampnames_data.extend(temp_sampnames)
    chrs_data.extend(temp_chrs)
    readdepths_data.extend(temp_readdepths)
    start_inds_data.extend(temp_start_inds)
    end_inds_data.extend(temp_end_inds)

    lens = [len(v) for v in readdepths_data]

    lengthfilter = [True if v < 2005  else False for v in lens]

    sampnames_data = np.array(sampnames_data)[lengthfilter]
    chrs_data = np.array(chrs_data)[lengthfilter]
    readdepths_data = np.array(readdepths_data)[lengthfilter]
    start_inds_data = np.array(start_inds_data)[lengthfilter]
    end_inds_data = np.array(end_inds_data)[lengthfilter]

    readdepths_data = np.array([np.array(k) for k in readdepths_data])
    readdepths_data = sequence.pad_sequences(readdepths_data, maxlen=2004,dtype=np.float32,value=-1)
    readdepths_data = readdepths_data[ :, None, :]

    test_x = torch.FloatTensor(readdepths_data)
    test_x = TensorDataset(test_x)
    x_test = DataLoader(test_x, batch_size=bs)

    allpreds = []
    with torch.no_grad():
        for segment in x_test:

            segment = segment[0].to(device)
            segment[:,:,:1000] -= means_
            segment[:,:,:1000] /=  stds_

            mask = torch.logical_and(segment != -1, segment != 0)

            mask = torch.squeeze(mask)
            real_mask = torch.ones(segment.size(0),2005, dtype=torch.bool).to(device)
            real_mask[:,1:] = mask

            output1 = model(segment,real_mask)

            _, predicted = torch.max(output1.data, 1)
            
            preds = list(predicted.cpu().numpy().astype(np.int64))
            allpreds.extend(preds)        

    chrs_data = chrs_data.astype(int)
    allpreds = np.array(allpreds)
    print("allpreds: ", np.shape(allpreds))

    result = pd.DataFrame(columns=["chr", "start", "end", "prediction"])

    for j in range(1,25):
        indices = chrs_data == j

        predictions = allpreds[indices]
        start_inds = start_inds_data[indices]
        end_inds = end_inds_data[indices]

        sorted_ind = np.argsort(start_inds)
        predictions = predictions[sorted_ind]
     
        end_inds = end_inds[sorted_ind]
        start_inds = start_inds[sorted_ind]
        chr_ = "chr"
        if j < 23:
            chr_ = str(j)
        elif j == 23:
            chr_ = "X"
        elif j == 24:
            chr_ = "Y"

        for k_ in range(len(end_inds)):
            result = result.append({
                "chr": chr_,
                "start": start_inds[k_],
                "end": end_inds[k_],
                "prediction": predictions[k_]
            }, ignore_index=True)


    # CNV fix
    label_raw = result['prediction'].copy()
    chr_raw = result['chr'].copy()
    label_fixed = label_raw.copy()

    for i in range(2, len(label_raw) - 2):
        if label_raw[i] == 0:
            if (label_raw[i-2] == label_raw[i-1] == 1) and (label_raw[i+1] == label_raw[i+2] == 1) and (chr_raw[i-2] == chr_raw[i-1] == chr_raw[i] == chr_raw[i+1] == chr_raw[i+2]):
                label_fixed[i] = 1
            elif (label_raw[i-2] == label_raw[i-1] == 2) and (label_raw[i+1] == label_raw[i+2] == 2) and (chr_raw[i-2] == chr_raw[i-1] == chr_raw[i] == chr_raw[i+1] == chr_raw[i+2]):
                label_fixed[i] = 2

    result['prediction'] = label_fixed
    result['prediction'] = result['prediction'].replace({0: 'nocall', 1: 'deletion', 2: 'duplication'})
    output_path = os.path.join(cur_dirname, "../", args.output, sample_name + "_segment.bed")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False, header=False)


    #CNV merge
    mask = (result.iloc[:, -1] == 'deletion') | (result.iloc[:, -1] == 'duplication')
    data_new = result[mask]
    
    merged_data = pd.DataFrame(columns=['chr', 'start', 'end', 'prediction'])
    binsize = 100000
    index = 0
    while index < data_new.shape[0]:
        now_row = data_new.iloc[index]
        next_index = index + 1
        if next_index >= data_new.shape[0]:
            merged_data = pd.concat([merged_data, pd.DataFrame([now_row])], ignore_index=True)
            break
        next_row = data_new.iloc[next_index]
        # 合并相邻行
        while (now_row['end'] + binsize == next_row['end'] and now_row['prediction'] == next_row['prediction'] and now_row['chr'] == next_row['chr']):
            now_row = {'chr': now_row['chr'], 'start': now_row['start'], 'end': next_row['end'], 'prediction': now_row['prediction']}
            next_index += 1
            if next_index >= data_new.shape[0]:
                break
            next_row = data_new.iloc[next_index]
        merged_data = pd.concat([merged_data, pd.DataFrame([now_row])], ignore_index=True)
        index = next_index
    #merged_data['prediction'] = merged_data['prediction'].replace({1: 'deletion', 2: 'duplication'})
    merged_data['length'] = merged_data['end'] - merged_data['start']
    output_path = os.path.join(cur_dirname, "../", args.output, sample_name + "_cnv.bed")
    merged_data.to_csv(output_path, sep="\t", index=False,)

    del sampledata, temp_sampnames, temp_chrs, temp_start_inds, temp_end_inds, temp_readdepths
    gc.collect()

message("call CNV complete!")
