### Introduction
This repository is an implementation for a submission to kannada mnist classification
using a semi supervised learning. The current solution first trains on labelled data up until 20 epochs,
then it switches to semi supervised learning by training on selected high confident pseudo labels on
unlabelled testing data. 



### Installation and Usage
This repository is based on PyTorch 1.4 and it needs numpy, os, maths, random, pandas libraries.
To train the model with default hyper parameters:

   ```shell
   cd kannada_exp
   python run.py --path 'path/to/your/kannada_dataset' 
   ```

### Some attempts I tried but somehow didn't improve results:
1. large model including vgg and resent18, resnet80, however, it didn't improve the validation accuracy

2. Complicated data augmentation could even bring the performance down 
if not used carefully with this data set. Also they are pretty slow, so the current code doesn't
include data augmentation apart from normalisation. I tried both torchvision and my own augmentation implementation such as cutout

3. Ensemble improved validation acc on Dig-MNIST to 84%, however it doesn't help
with the test somehow. I tried ensemble on different model architectures and different
model capacities.

4. I also tried other regularisation such as consistency on different views of input, et al.


### Some issues I encountered:
1. I was using Dig-MNIST for validation so I didn't realise until very late 
that the real testing acc is much higher than the validation acc. 
This might imply the labels in Dig-MNIST might be wrong (e.g. seems like those are
acquired from non-native speakers)

