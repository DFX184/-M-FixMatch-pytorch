<!-- @format -->

You can configure the parameters to be adjusted in the `config.py` file, or simply embed `trainer.py` into your project. In `config.py` you need to configure the following settings:

```py

seed = 2022108
epochs  = 30
batch_size = 64
learning_rate = 0.003
resume_path = None
fix_params = {
    'threshold':0.9,
    'mu':7,
    'lambda':1,
    'T':1
}
# labeled data num
## dataset ['mnist','cifar10','stl10','pets']
eval_iters = 100
dataset = "mnist"
num_labeled = 100
root_path   = r'./data'
num_classes = 5


```

And we saved a pre-trained model **resnet-18(accuracy:70.2)** in `/PreTrainModel`.

run

```
python main.py
```
