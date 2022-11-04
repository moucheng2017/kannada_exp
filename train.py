from models import *
from helpers import *


def trainer(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # get mean, std of data:
    train_data_path, val_data_path, test_data_path, result_data_path = args.path
    train = pd.read_csv(train_data_path)
    val = pd.read_csv(val_data_path)
    test = pd.read_csv(test_data_path)

    train_x = train.iloc[:, 1:].values / 255.
    train_mean, train_std = np.nanmean(train_x), np.nanstd(train_x)

    val_x = val.iloc[:, 1:].values / 255.
    val_mean, val_std = np.nanmean(val_x), np.nanstd(val_x)

    test_x = test.iloc[:, 1:].values / 255.
    test_mean, test_std = np.nanmean(test_x), np.nanstd(test_x)

    # Training datasets and dataloaders:
    train_dataset = DatasetKMNIST(images_path=train_data_path,
                                  labels_path=train_data_path,
                                  mean=train_mean,
                                  std=train_std)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    # Validation datasets and dataloaders:
    val_dataset = DatasetKMNIST(images_path=val_data_path,
                                labels_path=val_data_path,
                                mean=val_mean,
                                std=val_std)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, drop_last=True)

    # Define training optimizer:
    network = Vgglight().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=4, verbose=False,
                                                           mode='max', threshold=0.0001)

    # training process:
    for j in range(args.epochs):

        network.train()
        running_loss = 0.0
        train_acc = 0.0
        counter_t = 0

        # alpha as a hyper parameter for consistency regularisation
        if j <= 5:
            alpha = 0
        else:
            alpha = min(j * (1 / epochs), 0.1)

        for i, (inputs_s, inputs_w, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            counter_t += 1
            inputs_s, inputs_w, labels = inputs_s.cuda(), inputs_w.cuda(), labels.cuda()
            outputs_w, outputs_s = network(inputs_w), network(inputs_s)
            pred_w = outputs_w.argmax(dim=1)
            loss = 0.5 * criterion(outputs_w, labels) + 0.5 * criterion(outputs_s, labels)
            loss += alpha * criterion(outputs_s, pred_w.long()) # fix-match style consistency regularisation
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct = preds.eq(labels.view_as(pred_w)).sum().item()
            train_acc += correct / inputs_s.size()[0]
        running_loss = running_loss / counter_t
        train_acc = train_acc / counter_t

        # evaluation:
        network.eval()
        val_acc = 0
        counter_v = 0
        with torch.no_grad():
            for (_, data, target) in val_loader:
                counter_v += 1
                data, target = data.cuda(), target.cuda()
                output = network(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                val_acc += correct / data.size()[0]
        val_acc = 100 * val_acc / counter_v
        scheduler.step(val_acc)
        print('[epoch %d] loss: %.4f, train acc:% 4f, val acc: %.4f' % (j + 1, running_loss, train_acc, val_acc))
    print('Finished Training\n')

    # testing:
    network.eval()
    predictions = []
    test_x = np.reshape(test_x, (-1, 1, 28, 28))
    test_x_o = np.shape(test_x)[0]
    for i in range(test_x_o):
        data = np.expand_dims(test_x[i, :, :, :], axis=0)
        data = resize(data, (1, 1, 32, 32))
        data = (data - test_mean) / test_std
        data = torch.from_numpy(data).type(torch.FloatTensor).to('cuda')
        pred = network(data)
        pred = pred.max(dim=1)[1]
        predictions += list(pred.data.cpu().numpy())

    print('Testing is done')

    # write down the results:
    submission = pd.read_csv(result_data_path)
    submission['label'] = predictions
    submission.to_csv('submission.csv', index=False)
    submission.head()

    print('Writing into results, done.')

