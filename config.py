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