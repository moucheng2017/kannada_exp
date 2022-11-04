from train import *


def main():
    parser = argparse.ArgumentParser('training on kannada mnist', add_help=False)
    parser.add_argument('--seed', '-s', default=2022, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--path', '-p', default=None, type=str, help='Path to the folder containing all of the data in csv')

    global args
    args = parser.parse_args()
    trainer(args)


if __name__ == '__main__':
    main()