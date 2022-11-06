import random

from torchvision import transforms
import torchvision
from models import *
from dev_history.Models_resnet152 import ResNet50
from helpers import *


def cutout(image):
    patch_h = np.random.randint(0, 14)
    patch_w = np.random.randint(0, 14)

    h0 = np.random.randint(0, 28 - patch_h)
    w0 = np.random.randint(0, 28 - patch_w)
    image[:, :, h0:h0 + patch_h, w0:w0 + patch_w] = 0.0
    return image


strong_augmentation = torch.nn.Sequential(
    transforms.RandomRotation((-60, 60)),
    transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.4), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, shear=10)
)
strong_transforms = torch.jit.script(strong_augmentation)


weak_augmentation = torch.nn.Sequential(
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomHorizontalFlip(0.5)
)
weak_transforms = torch.jit.script(weak_augmentation)


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
    if args.network == 'resnet':
        network = ResNet50(10, 1).cuda()
    else:
        network = Net(0.5).cuda()  # dropout ratio 0.5

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(network.parameters(), lr=args.lr)
    steps_each_epoch = 60000 // args.batch
    total_steps = steps_each_epoch*args.epochs
    warmup_steps = int(args.warmup*total_steps)
    best_val_acc = 0.0

    for j in range(total_steps):

        if j < steps_each_epoch * args.ssl_start:
            alpha_current = 0
        elif j < warmup_steps:
            alpha_current = min(args.alpha, args.alpha * (j - steps_each_epoch * args.ssl_start) / (warmup_steps - steps_each_epoch * args.ssl_start))
        else:
            alpha_current = args.alpha

        network.train()

        try:
            images, labels = next(train_iterator)
            images_u, _ = next(test_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            test_iterator = iter(test_loader)
            images, labels = next(train_iterator)
            images_u, _ = next(test_iterator)

        optimizer.zero_grad()

        with torch.no_grad():

            if args.sup_aug == 1:
                if random.random() >= 0.5:
                    images = weak_augmentation(images)
                else:
                    images = strong_augmentation(images)

                if random.random() >= 0.5:
                    images = cutout(images)

            if args.unsup_aug == 1:
                images_u_s = strong_augmentation(images_u)
                images_u_s = cutout(images_u_s)

                images_u_w = weak_augmentation(images_u)

            else:
                images_u_w = images_u
                images_u_s = cutout(images_u)

            outputs_u_w = network(images_u_w)  # output of unlabelled data original
            pseudo_labels_soft = torch.softmax(outputs_u_w.detach() / 2.0, dim=-1)
            prob, pseudo_labels = torch.max(pseudo_labels_soft, 1, keepdim=False)
            mask = prob.ge(0.95).float()  # we only use high confident predictions as pseudo labels

        outputs = network(images)
        loss = criterion(outputs, labels) # superivsed learning on labelled data
        outputs_u = network(images_u_s)  # output of unlabelled data original after augmentation
        pseudo_loss = alpha_current*(criterion(outputs_u * mask.unsqueeze(1), (pseudo_labels * mask).long())).mean() # unsupervised learning
        loss += pseudo_loss

        loss.backward()
        optimizer.step()

        if args.lr_decay == 1:
            optimizer.param_groups[0]['lr'] = args.lr * (1 - j / total_steps) ** 0.99
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = args.lr

        preds = outputs.argmax(dim=1, keepdim=True)
        correct = preds.eq(labels.view_as(preds)).sum().item()
        train_acc = correct / images.size()[0]
        running_loss = loss.item()
        train_acc = 100 * train_acc

        if j % 100 == 0:
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
            if val_acc > best_val_acc:
                # save model
                torch.save(network.state_dict(), 'model.pt')
                # update best val acc
                best_val_acc = max(best_val_acc, val_acc)

    print('Finished Training\n')

    # last evaluation:
    if args.network == 'resnet':
        bestnetwork = ResNet50(10, 1).cuda()
    else:
        bestnetwork = Net(0.5).cuda()  # dropout ratio 0.5
    bestnetwork.load_state_dict(torch.load('model.pt'))
    bestnetwork.eval()
    val_acc = 0
    counter_v = 0
    with torch.no_grad():
        for (v_img, v_target) in val_loader:
            counter_v += 1
            v_output = bestnetwork(v_img)
            v_pred = v_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            v_correct = v_pred.eq(v_target.view_as(v_pred)).sum().item()
            val_acc += v_correct / v_img.size()[0]
    val_acc = 100 * val_acc / counter_v
    print('Final val acc: %.4f' % val_acc)

    # testing:
    # network.eval()
    predictions = []
    test_x_o = np.shape(test_x)[0]
    for i in range(test_x_o):
        data = np.expand_dims(test_x[i, :, :, :], axis=0)
        data = torch.from_numpy(data).type(torch.FloatTensor).to('cuda')
        pred = bestnetwork(data)
        pred = pred.max(dim=1)[1]
        predictions += list(pred.data.cpu().numpy())

    print('Testing is done')

    # write down the results:
    submission = pd.read_csv(result_data_path)
    submission['label'] = predictions
    submission.to_csv('submission.csv', index=False)
    submission.head()

    print('Writing into results, done.')

