# RMCosSSL and RLMSoftmaxSSL losses for training semi-supervied taxonomy model based on GAN

## Relativistic Marin Cosine semi-supervised Learning (RMCosSSL)
- Move to `rmcos-gan-ssl` folder
- Train the model when changing the amount of labeled data [50, 100, 600, 1000, 3000]

  `python run.py --epoch 40 --change_label True`

- Train the model when changing the values of alpha [0.1, 0.3, 0.5, 0.7, 0.9]

  `python run.py --epoch 40 --change_alpha True`

## Relativistic Large Marin Softmax semi-supervised Learning (RLMSoftmaxSSL)
- Move to `rlmsoftmax-gan-ssl` folder
- Train the model when changing the amount of labeled data [50, 100, 600, 1000, 3000]

  `python run.py --epoch 40 --change_label True`
  
- Train the model when changing the values of alpha [0.1, 0.3, 0.5, 0.7, 0.9]

  `python run.py --epoch 40 --change_alpha True`

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Chạy RMCos-SSL cho Cifar10

Vào rmcos-ssl-cifar10, tạo một môi trường ảo, cài python 3.9

conda istall pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install -r install.txt

bash data-local/bin/prepare_cifar10.sh


bash single_gpu.sh


