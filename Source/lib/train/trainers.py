from .base_trainer import BaseTrainer

trainers={
            "base":BaseTrainer,
}

def get_trainer(tag,default="base"):
    tag=tag.lower().replace("_","")
    if isinstance(tag,str) and tag in trainers.keys():
        return trainers[tag]
    else:
        return trainers[tag.lower().replace("_","")]
