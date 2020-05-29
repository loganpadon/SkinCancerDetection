# Separates the database into an 80-20 split database
from shutil import move
from random import seed, randint
import pandas as pd
seed(333)
def get_data_frame(dir):
    df = pd.read_csv(dir)
    df = df[['image_id','dx']]
    return df

spreadsheet_dir = 'data/HAM10000_metadata.csv'
df = get_data_frame(spreadsheet_dir)

for _,img,dx in df.itertuples():
    if randint(0, 100) <= 80: # Creates the 80-20 split
        subD = 'train'
    else:
        subD = 'test'
    orig = 'data/imgs/{}.jpg'.format(img)
    end = 'data/imgs/{}/{}/{}.jpg'.format(subD, dx, img)
    move(orig, end)