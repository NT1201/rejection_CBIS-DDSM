import os, time
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, root_split_dir, size=224):
        self.root = root_split_dir
        self.df = pd.read_csv(os.path.join(root_split_dir, "labels.csv"))
        self.t = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        rel = self.df.iloc[i]["image"]
        p = os.path.join(self.root, rel)
        x = Image.open(p).convert("L")
        x = self.t(x.convert("RGB"))
        y = int(self.df.iloc[i]["label"])
        return x, y, rel

root = r"C:/ThisMyFinal/cbis_cls3"
ds = CSVDataset(os.path.join(root, "train"))
print("train samples:", len(ds))
ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
t0=time.time(); n=0
for i,(x,y,rel) in enumerate(ld):
    n += x.size(0)
    if (i+1)%50==0:
        print(f"batches: {i+1}, images read: {n}, elapsed: {time.time()-t0:.1f}s")
        break
print("OK: loader can read batches.")
