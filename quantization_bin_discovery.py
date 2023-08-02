import csv
import torch
import pandas as pd
from glob import glob

def write_quantiles_tocsv(csv_name):

    quantiles = torch.arange(0.0, 1.01, 0.125)
    paths = glob("Data_generation/slf_mat/*.pt")


    with open(csv_name, 'w') as f:
        writer = csv.writer(f)

        for idx, pth in enumerate(paths):
            print(f"{idx} bins discovered.")
            data = torch.log( torch.load(pth).reshape(-1,1) + 1e-6 )
            qts = torch.quantile(data, quantiles)
            writer.writerow( [pth.split("/")[-1]] + qts.tolist() )
            

def find_quantization_bins(csv_name):
    
    df = pd.read_csv(csv_name, header=None)
    print(df.mean(numeric_only = True))
    print(df.median(numeric_only = True))
    
    
    
    
if __name__=="__main__":
    
    # write_quantiles_tocsv("quantile_data.csv")
    find_quantization_bins("quantile_data.csv")