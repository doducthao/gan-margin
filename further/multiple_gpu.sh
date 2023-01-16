# cifar10
# rlmsoftmax

# python train.py --labels data-local/labels/cifar10/1000_balanced_labels/00.txt --epochs 300 --mode rlmsoftmax

# python train.py --labels data-local/labels/cifar10/1000_balanced_labels/01.txt --epochs 300 --mode rlmsoftmax

# python train.py --labels data-local/labels/cifar10/1000_balanced_labels/02.txt --epochs 300 --mode rlmsoftmax

# python train.py --labels data-local/labels/cifar10/1000_balanced_labels/03.txt --epochs 300 --mode rlmsoftmax

# python train.py --labels data-local/labels/cifar10/1000_balanced_labels/04.txt --epochs 300 --mode rlmsoftmax

# python train.py --labels data-local/labels/cifar10/1000_balanced_labels/05.txt --epochs 300 --mode rlmsoftmax

# python train.py --labels data-local/labels/cifar10/4000_balanced_labels/00.txt --epochs 300 --mode rlmsoftmax

# python train.py --labels data-local/labels/cifar10/4000_balanced_labels/01.txt --epochs 300 --mode rlmsoftmax

# python train.py --labels data-local/labels/cifar10/4000_balanced_labels/02.txt --epochs 300 --mode rlmsoftmax --resume out_rlmsoftmax/cifar10/4000/02/2022-08-05_17:02:20

# python train.py --labels data-local/labels/cifar10/4000_balanced_labels/03.txt --epochs 300 --mode rlmsoftmax --resume out_rlmsoftmax/cifar10/4000/03/2022-08-09_09:06:31

# python train.py --labels data-local/labels/cifar10/4000_balanced_labels/04.txt --epochs 300 --mode rlmsoftmax

# python train.py --labels data-local/labels/cifar10/4000_balanced_labels/05.txt --epochs 300 --mode rlmsoftmax

# rmcos

python train.py --labels data-local/labels/cifar10/1000_balanced_labels/00.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256 --resume out_rmcos/cifar10/1000/00/2022-08-11_17:24:59

python train.py --labels data-local/labels/cifar10/1000_balanced_labels/01.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256

python train.py --labels data-local/labels/cifar10/1000_balanced_labels/02.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256

python train.py --labels data-local/labels/cifar10/1000_balanced_labels/03.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256

python train.py --labels data-local/labels/cifar10/1000_balanced_labels/04.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256

python train.py --labels data-local/labels/cifar10/4000_balanced_labels/00.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256

python train.py --labels data-local/labels/cifar10/4000_balanced_labels/01.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256

python train.py --labels data-local/labels/cifar10/4000_balanced_labels/02.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256

python train.py --labels data-local/labels/cifar10/4000_balanced_labels/03.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256

python train.py --labels data-local/labels/cifar10/4000_balanced_labels/04.txt --epochs 300 --mode rmcos --batch-size 256 --generated-batch-size 256
