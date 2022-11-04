import argparse
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import StepLR
import os
from torchcontrib.optim import SWA


from Helpers import *
from Model_cnn import *


def train_full(args):

    # reproducibility:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get data:
    train_data_path, val_data_path, test_data_path = get_data_full_path(args.path)
    train_x, train_y, val_x, val_y, test_x, test_id = get_data_values(args.path)
    x_mean, x_std = calculate_mean_std(train_x, val_x, test_x)

    # normalise images:
    train_x = (train_x - x_mean + 1e-8) / (x_std + 1e-8)
    val_x = (val_x - x_mean + 1e-8) / (x_std + 1e-8)
    test_x = (test_x - x_mean + 1e-8) / (x_std + 1e-8)

    # custom data set with data augmentation for example
    # train_dataset = DatasetKMNIST(images_path=train_data_path,
    #                               labels_path=train_data_path,
    #                               augmentation_gaussian=args.augmentation_gaussian,
    #                               augmentation_contrast=args.augmentation_contrast,
    #                               augmentation_zoom=args.augmentation_zoom,
    #                               augmentation_cutout=args.augmentation_cutout,
    #                               mean=x_mean,
    #                               std=x_std)

    torch_train_x = torch.from_numpy(train_x).type(torch.FloatTensor).to('cuda')
    torch_train_y = torch.from_numpy(train_y).type(torch.FloatTensor).to('cuda')
    train_dataset = torch.utils.data.TensorDataset(torch_train_x, torch_train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    # val data loader
    torch_val_x = torch.from_numpy(val_x).type(torch.FloatTensor).to('cuda')
    torch_val_y = torch.from_numpy(val_y).type(torch.FloatTensor).to('cuda')
    val_dataset = torch.utils.data.TensorDataset(torch_val_x, torch_val_y)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, drop_last=False)

    # define three networks for ensembling:
    # depth_list = [int(item) for item in args.depth_list.split(',')]
    # width_list = [int(item) for item in args.width_list.split(',')]
    # kernel_list = [int(item) for item in args.kernel_list.split(',')]
    # net1 = CNN(n_classes=10, input_dim=1, width=width_list[0], depth=depth_list[0], kernel=kernel_list[0], dropout_ratio=0.5).to('cuda')
    # net2 = CNN(n_classes=10, input_dim=1, width=width_list[1], depth=depth_list[1], kernel=kernel_list[1], dropout_ratio=0.5).to('cuda')
    # net3 = CNN(n_classes=10, input_dim=1, width=width_list[2], depth=depth_list[2], kernel=kernel_list[2], dropout_ratio=0.5).to('cuda')

    net1 = Net(dropout=0.3).to('cuda')
    net2 = Net(dropout=0.5).to('cuda')
    net3 = Net(dropout=0.7).to('cuda')

    # define loss function:
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    # define optimizers:
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9)
    optimizer3 = optim.SGD(net3.parameters(), lr=args.lr, momentum=0.9)

    # define learning rate schedulers:
    scheduler1 = StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler2 = StepLR(optimizer2, step_size=args.step_size, gamma=args.gamma)
    scheduler3 = StepLR(optimizer3, step_size=args.step_size, gamma=args.gamma)

    # train three networks
    v_acc1, net1 = train(epochs=args.epochs, network=net1, criterion=criterion, train_loader=train_loader, optimizer=optimizer1, scheduler=scheduler1, val_loader=val_loader, network_flag=1)
    v_acc2, net2 = train(epochs=args.epochs, network=net2, criterion=criterion, train_loader=train_loader, optimizer=optimizer2, scheduler=scheduler2, val_loader=val_loader, network_flag=2)
    v_acc3, net3 = train(epochs=args.epochs, network=net3, criterion=criterion, train_loader=train_loader, optimizer=optimizer3, scheduler=scheduler3, val_loader=val_loader, network_flag=3)
    print('val acc (net1): %.4f, val acc (net2): %.4f, val acc (net3): %.4f,' % (v_acc1, v_acc2, v_acc3))

    # ensemble evaluation
    net1.eval()
    net2.eval()
    net3.eval()
    acc = 0
    counter = 0
    with torch.no_grad():
        for data, target in val_loader:
            counter += 1
            data, target = data.to('cuda'), target.to('cuda')
            output1 = net1(data)
            output2 = net2(data)
            output3 = net3(data)
            output = (output1 + output2 + output3) / 3
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc += correct / data.size()[0]
    val_acc = 100 * acc / counter
    print('Accuracy of the network %.4f' % val_acc)
