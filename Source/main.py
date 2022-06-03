from lib.setup import Setup
from train import trainer
from test import tester
#from demo import demo

tasks={
        "train":trainer,
        "test":tester,
        #"demo":demo,
        }

if __name__=='__main__':
    setup = Setup().parse()
    task=tasks[setup.purpose]
    task(setup)
