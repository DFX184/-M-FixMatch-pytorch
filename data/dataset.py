from torch.utils.data import Dataset
from torchvision.datasets import STL10, CIFAR10, MNIST
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.transforms.transforms import RandomAffine
from .randaugument import*
from torchvision.datasets import DatasetFolder
import os

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)


class TransformFixMatch(object):
    def __init__(self, mean, std, norm=False):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=96,
                                  padding=4,
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=96,
                                  padding=4,
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        if norm:
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        else:
            self.normalize = transforms.ToTensor()

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransTwo:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return (self.transform(x), self.transform(x))


def get_transforms(num_size: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(num_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(stl10_mean, stl10_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(num_size),
        transforms.ToTensor(),
        transforms.Normalize(stl10_mean, stl10_std)
    ])
    return transform_train, transform_val


transform_train_cifar, transform_val_cifar = get_transforms(32)


def build_cifar_dataset(num_labeled):

    train_cifar_dataset = datasets.CIFAR10(
        root="./cifar10", download=True, train=True, transform=TransformFixMatch(stl10_mean, stl10_std))

    test_cifar_dataset = datasets.CIFAR10(
        root='./cifar10', download=True, train=False, transform=transform_val_cifar)

    splits = [num_labeled, len(train_cifar_dataset) - num_labeled]

    labeled_dataset, unlabeled_dataset = random_split(
        train_cifar_dataset, splits)

    return labeled_dataset, unlabeled_dataset, test_cifar_dataset


# Perform random crops and mirroring for data augmentation

transform_train, transform_val = get_transforms(96)

# Load train and validation sets


def build_stl_dataset(unlabeled_num: int, labeled_frac=0.1):
    trainval_set = datasets.STL10(
        r"./stl_dataset", split='train', transform=transform_train, download=True)

    # Use 10% of data for training - simulating low data scenario
    num_train = int(len(trainval_set)*labeled_frac)

    train_set, val_set = random_split(
        trainval_set, [num_train, len(trainval_set)-num_train])

    test_set = datasets.STL10(r"./stl_dataset", split='test',
                              transform=transform_val, download=True)

    unlabelled_set = datasets.STL10(
        r"./stl_dataset", transform=TransformFixMatch(stl10_mean, stl10_std), split='unlabeled', download=True)

    unlabeled_set, _ = random_split(
        unlabelled_set, [unlabeled_num, len(unlabelled_set)-unlabeled_num])
    del unlabelled_set

    return train_set, val_set, test_set, unlabeled_set


def get_transforms_food(num_size: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(num_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),

    ])

    transform_val = transforms.Compose([
        transforms.Resize(num_size),
        transforms.ToTensor()
    ])
    return transform_train, transform_val


def build_food_dataset():
    root = r"F:\\DataSets\\food-11"
    transform_train, transform_val = get_transforms_food(96)

    train_set = DatasetFolder(os.path.join(root, "training/labeled"),
                              loader=lambda x: Image.open(x), extensions="jpg", transform=transform_train)
    valid_set = DatasetFolder(os.path.join(root, "validation"), loader=lambda x: Image.open(
        x), extensions="jpg", transform=transform_train)
    unlabeled_set = DatasetFolder(os.path.join(root, "training/unlabeled"), loader=lambda x: Image.open(
        x), extensions="jpg", transform=TransformFixMatch((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)))
    test_set = DatasetFolder(os.path.join(root, "testing"), loader=lambda x: Image.open(
        x), extensions="jpg", transform=transform_train)

    return train_set, valid_set, test_set, unlabeled_set


class MNISTSSL(MNIST):
    def __init__(self, root, classes=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.data = np.array(self.data)

        if classes != None:
            sum_indexs = []
            for c in classes:
                indexs = np.where((self.targets == c))[0]
                sum_indexs += indexs.tolist()
            sum_index = np.array(sum_indexs).astype('int')
            self.data = self.data[sum_indexs]
            self.targets = np.array(self.targets)[sum_indexs]
            assert np.unique(self.targets).tolist() == classes
            self.targets = LabelEncoder().fit_transform(self.targets)

    def set_data(self, data, target):
        self.data = data.copy()
        self.targets = target.copy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10SSL(CIFAR10):
    def __init__(self, root, classes=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

        if classes != None:
            sum_indexs = []
            for c in classes:
                indexs = np.where((self.targets == c))[0]
                sum_indexs += indexs.tolist()
            sum_index = np.array(sum_indexs).astype('int')
            self.data = self.data[sum_indexs]
            self.targets = np.array(self.targets)[sum_indexs]
            # print(np.unique(self.targets))
            assert np.unique(self.targets).tolist() == classes
            self.targets = LabelEncoder().fit_transform(self.targets)

    def set_data(self, data, target):
        self.data = data.copy()
        self.targets = target.copy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class STL10SSL(STL10):
    def __init__(self, root, classes=None, split="train",
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        if classes != None:
            sum_indexs = []
            for c in classes:
                indexs = np.where((self.labels == c))[0]
                sum_indexs += indexs.tolist()
            sum_index = np.array(sum_indexs).astype('int')
            self.data = self.data[sum_indexs]
            self.labels = np.array(self.labels)[sum_indexs]
            print(np.unique(self.labels))
            assert np.unique(self.labels).tolist() == classes
            self.labels = LabelEncoder().fit_transform(self.labels)

    def set_data(self, data, target):
        self.data = data.copy()
        self.labels = target.copy()

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_mnist_dataset(root,num_labeled=100, size=96):
    transform_labeled = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),

        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])

    datasets = MNISTSSL(root=root, classes=[0, 2, 5, 6, 9], train=True)
    data = datasets.data
    target = datasets.targets
    sum_len = len(data)
    labeled_size = num_labeled/sum_len
    Xunlabel, Xlabeled, target1, target2 = train_test_split(
        data, target, test_size=labeled_size)

    test_dataset = MNISTSSL(
        root=root, classes=[0, 2, 5, 6, 9], train=False, transform=transform_labeled)

    label_dataset = MNISTSSL(
        root=root, classes=[0, 2, 5, 6, 9], train=True, transform=transform_labeled)
    label_dataset.set_data(Xlabeled, target2)

    unlabel_dataset = MNISTSSL(root=root, classes=[0, 2, 5, 6, 9], train=True, transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        TransformFixMatch(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))]))
    unlabel_dataset.set_data(Xunlabel, target1)

    labeled_loader = DataLoader(
        label_dataset, shuffle=True, batch_size=64, drop_last=True)
    unlabeled_loader = DataLoader(
        unlabel_dataset, shuffle=True, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True,
                             batch_size=64, pin_memory=True, drop_last=True)
    return (labeled_loader, unlabeled_loader, test_loader)


def get_cifar_dataset(root,num_labeled=100, size=96):
    transform_labeled = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),

        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])

    datasets = CIFAR10SSL(root=root, classes=[0, 1, 5, 6, 7], train=True)
    data = datasets.data
    target = datasets.targets
    sum_len = len(data)
    labeled_size = num_labeled/sum_len
    Xunlabel, Xlabeled, target1, target2 = train_test_split(
        data, target, test_size=labeled_size)

    test_dataset = CIFAR10SSL(
        root=root, classes=[0, 1, 5, 6, 7], train=False, transform=transform_labeled)

    label_dataset = CIFAR10SSL(
        root=root, classes=[0, 1, 5, 6, 7], train=True, transform=transform_labeled)
    label_dataset.set_data(Xlabeled, target2)

    unlabel_dataset = CIFAR10SSL(root=root, classes=[0, 1, 5, 6, 7], train=True, transform=transforms.Compose([
        transforms.Resize((size, size)),
        TransformFixMatch(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))]))
    unlabel_dataset.set_data(Xunlabel, target1)

    labeled_loader = DataLoader(
        label_dataset, shuffle=True, batch_size=64, drop_last=True)
    unlabeled_loader = DataLoader(
        unlabel_dataset, shuffle=True, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True,
                             batch_size=64, pin_memory=True, drop_last=True)
    return (labeled_loader, unlabeled_loader, test_loader)


def get_stl10_dataset(root,num_labeled=100, size=96):
    transform_labeled = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),

        transforms.Normalize(mean=stl10_mean, std=stl10_std)])

    datasets = STL10SSL(
        root=root, classes=[1, 2, 3, 5, 8], split="train+unlabeled")
    print(len(datasets))
    data = datasets.data
    target = datasets.labels
    sum_len = len(data)
    labeled_size = num_labeled/sum_len
    Xunlabel, Xlabeled, target1, target2 = train_test_split(
        data, target, test_size=labeled_size)

    test_dataset = STL10SSL(
        root=root, classes=[1, 2, 3, 5, 8], split="test", transform=transform_labeled)

    label_dataset = STL10SSL(
        root=root, classes=[1, 2, 3, 5, 8], split="train", transform=transform_labeled)
    label_dataset.set_data(Xlabeled, target2)

    unlabel_dataset = STL10SSL(root=root, classes=[1, 2, 3, 5, 8], split="train", transform=transforms.Compose([
        transforms.Resize((size, size)),
        TransformFixMatch(mean=stl10_mean, std=stl10_std)]))
    unlabel_dataset.set_data(Xunlabel, target1)

    labeled_loader = DataLoader(
        label_dataset, shuffle=True, batch_size=64, drop_last=True)
    unlabeled_loader = DataLoader(
        unlabel_dataset, shuffle=True, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True,
                             batch_size=64, pin_memory=True, drop_last=True)
    return (labeled_loader, unlabeled_loader, test_loader)


class PetsDataset(Dataset):
    def __init__(self, root_path, classes=None, transform=None):
        self.classes = os.listdir(root_path) if classes == None else classes
        self.labelmap = {v: i for i, v in enumerate(self.classes)}
        self.data = []
        self.targets = []
        for category in self.classes:
            sub_path = os.path.join(root_path, category)
            for filename in os.listdir(sub_path):
                file_path = os.path.join(sub_path, filename)
                self.data.append(file_path)
                self.targets.append(self.labelmap[category])
        self.transform = transform

    def set_data(self, data, target):
        self.data = data
        self.targets = target

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        img = Image.open(image)
        if self.transform != None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def get_pets_dataset(root,num_labeled=100, test_size=0.2, size=96):
    transform_labeled = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)])
    pets = ['leonberger', 'great_pyrenees', 'wheaten_terrier', 'miniature_pinscher', 'saint_bernard', 'shiba_inu', 'yorkshire_terrier', 'Siamese',
            'japanese_chin', 'Russian_Blue', 'german_shorthaired', 'havanese', 'Maine_Coon', 'Bengal', 'american_pit_bull_terrier', 'Bombay', 'pug', 'chihuahua']
    datasets = PetsDataset(root_path=root, classes=pets)
    data = datasets.data
    target = datasets.targets
    sum_len = len(data)
    labeled_size = num_labeled/sum_len
    Xunlabel, Xlabeled, target1, target2 = train_test_split(
        data, target, test_size=labeled_size)

    unlabel_data, test_data, unlabel_target, test_target = train_test_split(
        Xunlabel, target1, test_size=test_size)

    test_dataset = PetsDataset(
        root_path=root, classes=pets, transform=transform_labeled)
    test_dataset.set_data(test_data, test_target)
    label_dataset = PetsDataset(
        root_path=root, classes=pets, transform=transform_labeled)

    label_dataset.set_data(Xlabeled, target2)

    unlabel_dataset = PetsDataset(root_path=root, classes=pets, transform=transforms.Compose([
        transforms.Resize((size, size)),
        TransformFixMatch(mean=stl10_mean, std=stl10_std)]))
    unlabel_dataset.set_data(unlabel_data, unlabel_target)

    labeled_loader = DataLoader(
        label_dataset, shuffle=True, batch_size=64, drop_last=True)
    unlabeled_loader = DataLoader(
        unlabel_dataset, shuffle=True, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True,
                             batch_size=64, pin_memory=True, drop_last=True)

    #print(len(label_dataset), len(unlabel_dataset), len(test_dataset))
    return (labeled_loader, unlabeled_loader, test_loader)
