import argparse
# Adapted from original INVASE repository


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_type',
        choices=['hb'],
        default='hb',
        type=str)
    parser.add_argument(
        '--learning_rate',
        help='learning rate of model training',
        default=0.001,
        type=float)
    parser.add_argument(
        '--workers',
        default=8,
        type=int,
        help="number of workers for the data loader"
    )
    parser.add_argument(
        '--max-epochs',
        default=300,
        type=int,
        help="number of workers for the data loader"
    )
    parser.add_argument(
        '--eval-freq',
        default=2,
        type=int,
        help="frequency of evaluations"
    )
    parser.add_argument(
        '--device',
        default="cuda",
        type=str,
        help="device to keep the sensors in",
        choices=["cpu", "cuda"]
    )

    parser.add_argument('--train_img', type=str,
                    default='./data/eye')
    # parser.add_argument('--train_img_baseline', type=str,
    #                 default='/home/cigit/disk1/hemoglobin/data/c_results')

    parser.add_argument('--train_label', type=str,
                    default='./data/train_sample.csv')
    parser.add_argument('--val_img', type=str,
                     default='./data/eye')
    # parser.add_argument('--val_img_baseline', type=str,
    #                 default='/home/cigit/disk1/hemoglobin/data/c_results')

    parser.add_argument('--val_label', type=str,
                    default='./data/val_sample.csv')
    
    parser.add_argument("--height", type=int, default=224, help='height of input image')
    parser.add_argument("--width", type=int, default=224, help='width of input image')

    parser.add_argument("--epochs", type=int, default=100, help='number of epochs to train')
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument('--nThread', type=int, default=16, help='number of threads for data loading')

    return parser.parse_args()
