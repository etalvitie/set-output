# set-output

- Python >= 3.6
- PyTorch Requirement >= 1.8.0

## Downloading & Running

### Environment Setup
```
pip install -r requirements.txt
pip install -e .
```

### Downloading Dataset
```
pip install gdown
gdown --id 1FbZe9EZ2847iMYVvQffpk_zLZO21pyvc # matched
gdown --id 15Olc5kRrV_HkFmOy3RN7vn2cVVcQQgH8 # raw
gdown --id 1gq028XAvreFu2Ynkxh7xOzfsIwk81IHY # bullet
```

### Running
```
python src\set_agent\minatar_dspn_model.py
```

### Viewing Loss Curve (Optional)
During or after a training session, visualize the loss using Tensorboard.
```
tensorboard --logdir=lightning_logs
```

## To-do List

[Notion](https://www.notion.so/bwww/LACE-Notes-ed219fd338f1413ba37ef293c6ea43c7)


## Set-DETR

Check this [Colab Notebook](https://colab.research.google.com/drive/1xZ6G2tlFeVxRQ1TNSmf4Cs1YoWkY8yhs?usp=sharing) for a minimum working example.
