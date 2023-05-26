import argparse
# Adapted from original INVASE repository


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_type',
        default='hemo',
        type=str)
    parser.add_argument(
        '--learning_rate',
        help='learning rate of model training',
        default=0.0001,
        type=float)
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help="number of workers for the data loader"
    )
    parser.add_argument(
        '--lamda',
        help='invase hyper-parameter lambda',
        default=0.1,
        type=float)
    parser.add_argument(
        '--max-epochs',
        default=2,
        type=int,
        help="number of workers for the data loader"
    )
    parser.add_argument(
        '--eval-freq',
        default=1,
        type=int,
        help="frequency of evaluations"
    )
    parser.add_argument(
        '--device',
        default="cpu",
        type=str,
        help="device to keep the sensors in",
        choices=["cpu", "cuda"]
    )

    parser.add_argument('--train_img', type=str,
                    default='D:/HGB_data/hemo/eye_train_3')
    parser.add_argument('--train_img_baseline', type=str,
                    default='D:/HGB_data/hemo/conjunctiva_train_3')
    parser.add_argument('--train_label', type=str,
                    default='D:/HGB_data/hemo/train_3_class.csv')                 
    parser.add_argument('--val_img', type=str,
                    default='D:/HGB_data/hemo/eye_val_3')                
    parser.add_argument('--val_img_baseline', type=str,
                    default='D:/HGB_data/hemo/conjunctiva_val_3')               
    parser.add_argument('--val_label', type=str,
                    default='D:/HGB_data/hemo/val_3_class.csv')

    parser.add_argument("--height", type=int, default=224, help='height of input image')
    parser.add_argument("--width", type=int, default=224, help='width of input image')

    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument('--nThread', type=int, default=12, help='number of threads for data loading')

    return parser.parse_args()


def get_decode_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trained-path',
        required=True,
        type=str,
        help="path to the pretrained model",
    )
    parser.add_argument(
        '--decoder-epochs',
        default=10,
        type=int,
        help="number of epochs to tune the decoder",
    )
    parser.add_argument(
        '--device',
        default="cpu",
        type=str,
        help="device to keep the sensors in",
        choices=["cpu", "cuda"]
    )
    parser.add_argument(
        '--eval-freq',
        default=5,
        type=int,
        help="frequency of evaluations"
    )
    return parser.parse_args()