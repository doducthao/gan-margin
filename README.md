# GAN-SSL
- RMCos-SSL
- RLMSoftmax-SSL
- RMARC-SSL
## MNIST
cd to mnist/ and run

- ```bash retrain_margingan_mnist.sh``` 

to retrain marginGAN or
- ```bash multiple_gpu.sh``` 

to train gan-ssl.

## CIFAR10
cd to further/ and run
1. ```bash data-local/bin/prepare_cifar10.sh``` (prepare cifar10 dataset)
2. 
    - ```bash retrain_margingan_cifar10.sh```

    to retrain marginGAN or
    - ```bash multiple_gpu.sh``` 

    to train gan-ssl.

**Changing --mode in .sh file [rmcos, rlmsoftmax, rmarc, margingan] to play with losses**
