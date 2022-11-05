import argparse
from train import *


def main():
    parser = argparse.ArgumentParser('training on kannada mnist', add_help=False)
    parser.add_argument('--seed', '-s', default=1024, type=int, help='Random seed')
    parser.add_argument('--network', default='simplenet', type=str, help='resent or simplene')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--lr_decay', default=1, type=int, help='1 when decays the lr or 0 without decaying')
    parser.add_argument('--batch', default=1024, type=int, help='batch size of labelled data for training')
    parser.add_argument('--batch_test', default=128, type=int, help='batch size of unlabelled data for pseudo labelling')
    parser.add_argument('--epochs', default=150, type=int, help='total training steps')
    parser.add_argument('--path', '-p', default='/home/moucheng/projects_data/Kannada-MNIST', type=str, help='Path to the folder containing all of the data in csv')
    parser.add_argument('--alpha', default=0.01, type=float, help='weight on pseudo labelling loss')
    parser.add_argument('--ssl_start', default=40.0, type=float, help='epochs when we start to use ssl')
    parser.add_argument('--sup_aug', default=0, type=int, help='1 for using random aug on labelled data')
    parser.add_argument('--unsup_aug', default=1, type=int, help='1 for using random aug on unlabelled data')

    global args
    args = parser.parse_args()
    trainer(args)


if __name__ == '__main__':
    main()