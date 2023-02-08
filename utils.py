import torch
import torchvision
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split


LESS_DATA = 10000 # Int, >0 if less data should be used, otherwise 0
SERVER_TEST_SIZE = 1000
SERVER_TRAIN_SIZE = 100


def get_data_by_indices(name, train, indices):
    '''
    Returns the data of the indices.

    :param name: string, name of the dataset
    :param train: boolean, train or test data
    :param indices: list, indices of the data to use
    :return: dataset with the given indices
    '''
    if name == "CIFAR10": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif name == "MNIST": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif name == "FashionMNIST": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    elif name == "CIFAR100": # 100 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    else:
        raise NameError(f"No dataset named {name}. Choose from: CIFAR10, CIFAR100, MNIST, FashionMNIST")

    return torch.utils.data.Subset(dataset, indices)

# def split_data_by_indices(data, n, iid=True, shards_each=2):
#     #TODO: Have more influence on teh distribtution: classes per client, distribution via parameter
#     '''
#     Splits the given data in n splits, instances are represented by their indices. Splits can be either idd or not-iid, for the latter the parameter shards_each
#     will be used as well meaning the data will be sorted by class, splited in shards_each * n splits and #shards_each
#     will be randomly assigned to each client

#     :param data: Dataset
#     :param n: int, number of splits
#     :param iid: boolean, iid or non-iid splits
#     :param shards_each: int, see description
#     :return: list, containing the splits as indices
#     '''
#     data_size = LESS_DATA if LESS_DATA > 0 else len(data)


#     if iid:
#         local_len = np.floor(data_size / n)
#         total_n = int(local_len * n)

#         indices = torch.randperm(len(data)).numpy()[:total_n]
#         splits = np.split(indices, n)

#     else:
#         n_shards = n * shards_each
#         shard_len = int(np.floor(data_size / n_shards))
#         print(f'####shard len: {shard_len}')

#         indices = torch.randperm(len(data)).numpy()[:(n_shards * shard_len)]
#         targets = torch.Tensor(data.targets)
#         ind_targets = targets[indices]

#         sorted_indices = np.array([x for _, x in sorted(zip(ind_targets, indices))])

#         # print(f'####sorted indices: {sorted_indices}')
#         shards = np.split(sorted_indices, n_shards)
#         print(f'####shards: {len(shards)}')

#         random_shards = torch.randperm(n_shards)
#         shards = [shards[i] for i in random_shards]

#         splits = []
#         for i in range(0, len(random_shards), shards_each):
#             splits.append(np.concatenate(shards[i:i + shards_each]).tolist())

#     return splits


def get_non_idd_data(data, indices, targets, target_class, local_len, degree_niid):
    class_targets = indices[targets==target_class] # get indices of data beloning to target class
    # select random indices from this target class
    perm = torch.randperm(class_targets.size(0))
    client_indices = perm[:int(degree_niid * local_len)]
    client_indices = class_targets[client_indices].numpy()

    ## fill remaining spots with random datapoints, can be from any class
    perm = torch.randperm(len(data)).numpy() 
    client_indices_random = perm[:(local_len - int(degree_niid * local_len))]
    client_data_indices = np.concatenate((client_indices, client_indices_random)).tolist()

    return client_data_indices

def split_data_by_indices(data, n, iid=True, degree_niid=0.25):
    '''
    Splits the given data in n splits, instances are represented by their indices. Splits can be either idd or not-iid. For the first random data points will be assigned to each client.
    For the latter, the clients are divided into groups, where the number of groups are equal to the number of classes. Clients in each group are assigned a fixed fraction (degree_niid) of data from the designated class. 
    The rest of the data for the client are assigned randomly.

    :param data: Dataset
    :param n: int, number of splits
    :param iid: boolean, iid or non-iid splits
    :param degree_niid: float, degree of non-iid distribution. 0 is iid, while 1 is only one class for each client.
    :return: list, containing the splits as indices
    '''

    data_size = LESS_DATA if LESS_DATA > 0 else len(data)

    if iid:
        local_len = np.floor(data_size / n)
        total_n = int(local_len * n)

        indices = torch.randperm(len(data)).numpy()[:total_n]
        splits = np.split(indices, n)

    else:
        local_len = int(np.floor(data_size / n))
        
        indices = torch.tensor(np.arange((len(data)))) 
        targets = data.targets

        # get number of distinct classes in the dataset
        n_classes = targets.unique().size(0)

        splits = []
        n_class_assigned_clients = int(np.floor(n/n_classes))
        if n_class_assigned_clients == 0: # less clients than different classes in the data
            for i in range(0, n): 
                random_class = np.random.randint(n_classes)
                client_data_indices = get_non_idd_data(data, indices, targets, target_class = random_class, local_len = local_len, degree_niid = degree_niid)
                splits.append(client_data_indices)
        else: # more clients than different classes in data
            for i in range(0, n_classes): # loop through different classes
                for _ in range(0, n_class_assigned_clients): # loop through number of clients assigned to each class
                    client_data_indices = get_non_idd_data(data, indices, targets, target_class = i, local_len = local_len, degree_niid = degree_niid) # get data indices based on class and degree of non-iid
                    splits.append(client_data_indices)

            if len(splits) < n: # when there are still clients left that haven't been assigned a class: give them a random class. This happens when number of clients is not a multiple of number of classes
                for i in range(0, n - len(splits)):
                    random_class = np.random.randint(n_classes) 
                    client_data_indices = get_non_idd_data(data, indices, targets, target_class = random_class, local_len = local_len, degree_niid = degree_niid)
                    splits.append(client_data_indices)

    return splits

def split_data(data, n, iid=True, degree_niid=0.25):
    '''
    Does the same like split_data_by_indices but returns the data directly instead of indices.

    :param data: Dataset
    :param n: int, number of splits
    :param iid: boolean, iid or non-iid splits
    :param degree_niid: int, see description
    :return: list, containing the splits
    '''

    data_size = LESS_DATA if LESS_DATA > 0 else len(data)

    if iid:
        indices = torch.randperm(len(data))[:total_n]
        subset = torch.utils.data.Subset(data, indices)
        splits = torch.utils.data.random_split(subset, np.full(n, fill_value=local_len, dtype="int32"))

    else:
        local_len = int(np.floor(data_size / n))
        
        indices = torch.tensor(np.arange((len(data)))) 
        targets = data.targets

        # get number of distinct classes in the dataset
        n_classes = targets.unique().size(0)

        splits = []
        n_class_assigned_clients = int(np.floor(n/n_classes))

        if n_class_assigned_clients == 0: # less clients than different classes in the data
            for i in range(0, n): 
                random_class = np.random.randint(n_classes)
                client_data_indices = get_non_idd_data(data, indices, targets, target_class = random_class, local_len = local_len, degree_niid = degree_niid)
                splits.append(torch.utils.data.Subset(client_data_indices))
        else: # more clients than different classes in data
            for i in range(0, n_classes): # loop through different classes
                for _ in range(0, n_class_assigned_clients): # loop through number of clients assigned to each class
                    client_data_indices = get_non_idd_data(data, indices, targets, target_class = i, local_len = local_len, degree_niid = degree_niid) # get data indices based on class and degree of non-iid
                    splits.append(torch.utils.data.Subset(client_data_indices))

            if len(splits) < n: # when there are still clients left that haven't been assigned a class: give them a random class. This happens when number of clients is not a multiple of number of classes
                for i in range(0, n - len(splits)):
                    random_class = np.random.randint(n_classes) 
                    client_data_indices = get_non_idd_data(data, indices, targets, target_class = random_class, local_len = local_len, degree_niid = degree_niid)
                    splits.append(torch.utils.data.Subset(client_data_indices))
    return splits



# def split_data(data, n, iid=True, shards_each=2):
#     '''
#     Does the same like split_data_by_indices but returns the data directly instead of indices.

#     :param data: Dataset
#     :param n: int, number of splits
#     :param iid: boolean, iid or non-iid splits
#     :param shards_each: int, see description
#     :return: list, containing the splits
#     '''
#     local_len = np.floor(len(data) / n)
#     total_n = int(local_len * n)

#     if iid:
#         indices = torch.randperm(len(data))[:total_n]
#         subset = torch.utils.data.Subset(data, indices)
#         splits = torch.utils.data.random_split(subset, np.full(n, fill_value=local_len, dtype="int32"))

#     else:
#         n_shards = n * shards_each

#         sorted_indices = torch.argsort(data.targets)

#         shard_len = int(np.floor(len(data) / n_shards))
#         shards = list(torch.split(sorted_indices, shard_len))[:n_shards]

#         random_shards = torch.randperm(n_shards)
#         shards = [shards[i] for i in random_shards]

#         splits = []
#         for i in range(0, len(random_shards), shards_each):
#             splits.append(torch.utils.data.Subset(data, torch.cat(shards[i:i + shards_each])))

#     return splits

def get_data(name, train):
    '''
    Gets the corresponding data (either train or test data)
    :param name: string, name of the dataset
    :param train: boolean, train or test
    :return: dataset
    '''
    if name == "CIFAR10": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif name == "MNIST": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif name == "FashionMNIST": # 10 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
    elif name == "CIFAR100": # 100 classes
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    else:
        raise NameError(f"No dataset named {name}. Choose from: CIFAR10, CIFAR100, MNIST, FashionMNIST")

    if not train:
        indices = list(zip(np.arange(len(dataset)), dataset.targets))
        test_indices, train_indices = train_test_split(indices, test_size=SERVER_TEST_SIZE, train_size=SERVER_TRAIN_SIZE, shuffle=True, stratify=dataset.targets)
        test_indices, _ = zip(*test_indices)
        train_indices, _ = zip(*train_indices)
        dataset = (torch.utils.data.Subset(dataset, test_indices), torch.utils.data.Subset(dataset, train_indices))



    return dataset

class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


    


