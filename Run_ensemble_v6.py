from v6_ensemble_swa import *


def main():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--seed', '-s', default=2022, type=int, help='Random seed')

    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='exponential learning decay rate')
    parser.add_argument('--step_size', default=0.1, type=float, help='step size')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')

    parser.add_argument('--augmentation_gaussian', default=0, type=int, help='Gaussian noise data augmentation flag, 1 when use it and 0 when not use it')
    parser.add_argument('--augmentation_cutout', default=1, type=int, help='Cutout data augmentation flag, 1 when use it and 0 when not use it')
    parser.add_argument('--augmentation_zoom', default=0, type=int, help='Random Zoom in data augmentation flag, 1 when use it and 0 when not use it')
    parser.add_argument('--augmentation_contrast', default=0, type=int, help='Random Contrast in data augmentation flag, 1 when use it and 0 when not use it')

    parser.add_argument('--depth_list', default='5, 4, 3', type=str, help='Numbers of conv layers for 3 models')
    parser.add_argument('--width_list', default='64, 48, 32', type=str,help='Numbers of channels in the first conv layer for 3 modles')
    parser.add_argument('--kernel_list', default='3, 5, 7', type=str, help='Kernel sizes for 3 models')

    parser.add_argument('--path', '-p', default=None, type=str, help='Path to the folder containing all of the data in csv')

    global args
    args = parser.parse_args()

    train_full(args)


if __name__ == '__main__':
    main()