Set up environment:
```bash
conda create python=3.13 -n LFM4Ads
conda activate LFM4Ads
pip install torch -i https://download.pytorch.org/whl/cu121
pip install torcheval tqdm pandas pyarrow
```

Set up dataset:
```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz
md5sum KuaiRand-1K.tar.gz # 6b0b9c8222d67fcd4c676218edca3f1f
tar -xzvf KuaiRand-1K.tar.gz
python dataset.py
```

Run a trial:
```bash
python main.py cuda:0
```
Change `cuda:0` to `cpu` if no GPU.
The downstream AUCs will be saved in `result.csv`.
The AUCs in paper are averaged over 10 trials.
