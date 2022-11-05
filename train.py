from models import *
from helpers import *


def trainer(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data loaders
    train_data_path, val_data_path, test_data_path, result_data_path = get_data_full_path(args.path)
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess(train_data_path, val_data_path, test_data_path)
    train_loader, val_loader, test_loader = get_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y, args.batch, args.batch_test)
    train_iterator = iter(train_loader)
    test_iterator = iter(test_loader)

    # networks and optimizer:
    network = Net(0.5).cuda() # dropout ratio 0.5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(network.parameters(), lr=args.lr)
    steps_each_epoch = 60000 // args.batch
    warmup_steps = int(0.5*args.steps)
    for j in range(args.steps):

        if j < steps_each_epoch * 20:
            alpha_current = 0
        elif j < warmup_steps:
            alpha_current = min(args.alpha, j / warmup_steps)
        else:
            alpha_current = args.alpha

        network.train()
        running_loss = 0.0
        train_acc = 0.0
        counter_t = 0

        try:
            images, labels = next(train_iterator)
            images2, _ = next(test_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            test_iterator = iter(test_loader)
            images, labels = next(train_iterator)
            images2, _ = next(test_iterator)

        optimizer.zero_grad()
        counter_t += 1

        outputs = network(images)
        outputs2 = network(images2) # output of unlabelled data

        with torch.no_grad():
            pseudo_labels_soft = torch.softmax(outputs2.detach() / 2.0, dim=-1)
            prob, pseudo_labels = torch.max(pseudo_labels_soft, 1, keepdim=False)

        mask = prob.ge(0.95).float() # we only use high confident predictions as pseudo labels
        loss = criterion(outputs, labels) # superivsed learning on labelled data
        pseudo_loss = alpha_current*(F.cross_entropy(outputs2 * mask.unsqueeze(1), (pseudo_labels * mask).long(), reduction='mean')).mean() # unsupervised learning
        loss += pseudo_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        optimizer.param_groups[0]['lr'] = args.lr * (1 - j / args.steps) ** 0.99
        current_lr = optimizer.param_groups[0]['lr']

        preds = outputs.argmax(dim=1, keepdim=True)
        correct = preds.eq(labels.view_as(preds)).sum().item()
        train_acc += correct / images.size()[0]
        running_loss = running_loss / counter_t
        train_acc = 100 * train_acc / counter_t

        if j % steps_each_epoch == 0:
            # evaluation:
            network.eval()
            val_acc = 0
            counter_v = 0
            with torch.no_grad():
                for (v_img, v_target) in val_loader:
                    counter_v += 1
                    v_output = network(v_img)
                    v_pred = v_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    v_correct = v_pred.eq(v_target.view_as(v_pred)).sum().item()
                    val_acc += v_correct / v_img.size()[0]
            val_acc = 100 * val_acc / counter_v
            print('[step %d] loss: %.4f, lr: %.4f, train acc:%.4f, val acc: %.4f' % (j + 1, running_loss, current_lr, train_acc, val_acc))

    print('Finished Training\n')

    # testing:
    network.eval()
    predictions = []
    test_x_o = np.shape(test_x)[0]
    for i in range(test_x_o):
        data = np.expand_dims(test_x[i, :, :, :], axis=0)
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

