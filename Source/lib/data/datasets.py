from .base_dataset import BaseDataset

datasets={
            "base":BaseDataset,
}

def get_dataset(tag,default="base"):
    tag=tag.lower().replace("_","")
    if isinstance(tag,str) and tag in datasets.keys():
        return datasets[tag]
    else:
        return datasets[tag.lower().replace("_","")]
