from torch.utils.data import TensorDataset
from Models_resnet152 import *
from Dataset import DatasetKannadaMNIST

def args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--seed', '-s', default=2022, type=int, help='Random seed')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=0.9, type=float, help='exponential learning decay rate')
    parser.add_argument('--step_size', default=0.7, type=float, help='step size')
    parser.add_argument('--augmentation', default=1, type=int, help='data augmentation flag, 1 when use it and 0 when not use it')

    parser.add_argument('--epochs', '-e', default=20, type=int, help='Training epochs')
    # parser.add_argument('--depth', '-d', default=18, type=int, help='Depth of Network')
    # parser.add_argument('--width', '-w', default=32, type=int, help='Width of Network')
    parser.add_argument('--path', '-p', default=None, type=str, help='Path to the folder containing all of the data in csv')
    parser.add_argument('--device', '-de', default='gpu', type=str, help='device choice between gpu and cpu')
    return parser

def main(args):
    # read the files
    train_data_path = os.path.join(args.path, 'train.csv')
    val_data_path = os.path.join(args.path, 'Dig-MNIST.csv')
    test_data_path = os.path.join(args.path, 'test.csv')

    train = pd.read_csv(train_data_path)
    val = pd.read_csv(val_data_path)
    test = pd.read_csv(test_data_path)

    # split the data:
    train_x = train.iloc[:, 1:].values / 255.
    train_y = train.iloc[:, 0].values

    val_x = val.iloc[:, 1:].values / 255.
    val_y = val.iloc[:, 0].values

    test_x = test.iloc[:, 1:].values / 255.

    # reshape the data:
    train_x = np.reshape(train_x, (60000, 1, 28, 28))
    val_x = np.reshape(val_x, (10240, 1, 28, 28))
    test_x = np.reshape(test_x, (5000, 1, 28, 28))

    # calculate the mean and std:
    all_x = np.concatenate((train_x, val_x, test_x), axis=0)
    x_mean = np.nanmean(all_x)
    x_std = np.nanstd(all_x)

    # normalise images:
    train_x = (train_x - x_mean + 1e-8) / (x_std + 1e-8)
    val_x = (val_x - x_mean + 1e-8) / (x_std + 1e-8)
    test_x = (test_x - x_mean + 1e-8) / (x_std + 1e-8)

    # train data loader:
    if args.device == 'gpu':
        torch_train_x = torch.from_numpy(train_x).type(torch.FloatTensor).to('cuda')
        torch_train_y = torch.from_numpy(train_y).type(torch.LongTensor).to('cuda')
    else:
        torch_train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        torch_train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    # transform:
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomCrop(28),
    #     # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    #     transforms.RandomRotation(degrees=(0, 180)),
    #     transforms.ToTensor()
    # ])
    train_dataset = DatasetKannadaMNIST(images_path=train_data_path, labels_path=train_data_path, augmentation=args.augmentation, mean=x_mean, std=x_std)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    # train_dataset = torch.utils.data.TensorDataset(torch_train_x, torch_train_y)
    # train_dataset = DatasetKMNIST(images_path=train_data_path, labels_path=train_data_path, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    # val data loader
    if args.device == 'gpu':
        torch_val_x = torch.from_numpy(val_x).type(torch.FloatTensor).to('cuda')
        torch_val_y = torch.from_numpy(val_y).type(torch.FloatTensor).to('cuda')
    else:
        torch_val_x = torch.from_numpy(val_x).type(torch.FloatTensor)
        torch_val_y = torch.from_numpy(val_y).type(torch.FloatTensor)

    val_dataset = torch.utils.data.TensorDataset(torch_val_x, torch_val_y)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, drop_last=False)

    # test data loader:
    if args.device == 'gpu':
        torch_test_x = torch.from_numpy(test_x).type(torch.FloatTensor).to('cuda')
    else:
        torch_test_x = torch.from_numpy(test_x).type(torch.FloatTensor)

    test_dataset = torch.utils.data.TensorDataset(torch_test_x)


    net = ResNet152(10, 1).to('cuda')

    # if args.device == 'gpu':
    #     net = Net().to('cuda')
    # else:
    #     net = Net()

    # define loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    torch.manual_seed(args.seed)

    # train the network
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        counter_t = 0
        for i, data in enumerate(train_loader, 0):
            counter_t += 1
            inputs, labels = data
            inputs, labels = inputs.float().cuda(), labels.long().cuda()
            #inputs, labels = inputs, labels
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = preds.eq(labels.view_as(preds)).sum().item()
            train_acc += correct / inputs.size()[0]
        # print('[epoch %d] loss: %.3f' % (epoch + 1, running_loss / sampling_times))
        scheduler.step()
        train_acc = train_acc / counter_t

        # evaluation:
        net.eval()
        # correct = 0
        acc = 0
        counter_v = 0
        with torch.no_grad():
            for data, target in val_loader:
                counter_v += 1
                data, target = data.to('cuda'), target.to('cuda')
                # print(data.size())
                output = net(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                acc += correct / data.size()[0]
                # print('Accuracy of the network %d %%' % acc)
        # val_acc = 100 * correct / len(val_dataset)
        val_acc = 100 * acc / counter_v
        # print('Accuracy of the network %d %%' % val_acc)
        print('[epoch %d] loss: %.4f, train acc:% 4f, val acc: %.4f' % (epoch + 1, running_loss / counter_t, train_acc, val_acc))

    print('Finished Training\n')

    # validating
    # net.eval()
    # val = net(torch_val_x)
    # _, predicted = torch.max(val.data, 1)
    # # acc = 100 * torch.sum(torch_val_y == predicted) / len(torch_val_y)
    # # print('Accuracy of the network %d %%' % acc)
    # print('Accuracy of the network %d %%' % (100 * torch.sum(torch_val_y == predicted) / len(val_y)))

    # net.eval()
    # correct = 0
    # with torch.no_grad():
    #     for data, target in val_loader:
    #         data, target = data.to('cuda'), target.to('cuda')
    #         output = net(data)
    #         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #
    # val_acc = 100 * correct / len(val_dataset)
    # print('Accuracy of the network %d %%' % val_acc)

    # test:
    # net.eval()
    # predictions = []
    # test_x_o = np.shape(test_x)[0]
    #
    # for i in range(test_x_o):
    #     data = np.expand_dims(test_x[i, :, :, :], axis=0)
    #     data = torch.from_numpy(data).type(torch.FloatTensor).to('cuda')
    #     pred = net(data).max(dim=1)[1]
    #     predictions += list(pred.data.cpu().numpy())

    # print(predictions)
    # test_sample_path = os.path.join(args.path, 'sample_submission.csv')
    # submission = pd.read_csv(test_sample_path)
    # submission['label'] = predictions
    # submission.to_csv(test_data_path, index=False)
    # submission.head()
    #
    # return net


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # change the hyper parameters here:
    args.path = '/home/moucheng/projects_data/Kannada-MNIST'
    args.batch = 1024
    args.device = 'gpu'
    args.epochs = 30
    # args.epochs = 1
    args.seed = 1234
    args.lr = 0.001
    args.gamma = 0.7
    args.step_size = 80
    args.augmentation = 1

    main(args)