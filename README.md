# Convolutional neural network implementation on SVHN Dataset 

SVHN Dataset is a real world image dataset used for machine learning and object recognition. I have tried to achieve the SoTA performance on this dataset by using various methods. 

## Methods used to improve training and learning

I have used several data augmentation strategies that can be directly applied to the training images through PyTorch. They are:

* Random Horizontal Flip (Probability = 0.5) - This flips the image horizontally and the probability of the image being flipped is 0.5.
``` 
transforms.RandomHorizontalFlip()
```
* Gaussian Blur (kernel size = 3, sigma = (0.1, 0.2)) - This blurs the image by running a gaussian kernel of filter size 3 on the entire image. (You will need PyTorch 1.7 for this augmentation method to work)
Fun Fact - Did you know that adding Gaussian noise with zero mean to the input is equivalent to L2 weight decay when the loss function is mean square error.
```
transforms.GaussianBlur(kernel_size = 3, sigma=(0.1, 2.0))
```
* Color Jittering - This helps a lot, this method randomly changes the brightness and saturation of the image. 
```
transforms.ColorJitter(brightness=0.4, saturation=0.4)
```
* Normalizing, ofcourse on the image based on the mean and standard deviation of the dataset. 
```
transforms.Normalize((0.438, 0.444, 0.473), (0.195, 0.198, 0.195))
```
* I also used random rotation strategy to introduce some more variability in the training dataset.
```
transforms.RandomRotation(degrees=(10, -10))
```
* It also helped to increase the size of the image by using image interpolation techniques.
```
transforms.Resize(128, interpolation = 2)
```
and then to randomly crop it to the required size for the input using random resized crop.
```
transforms.RandomResizedCrop(64)
```

## Implementation 

* At first I implemented a simple CNN architecture with convolutional layers followed by a FC layer. It turns out that the dataset is too varied and the test accuracy was stuck at 70%. Check the SVHNNet class for the architecture in classification.ipynb

* Adding batch norm layers significantly improved the performance. Accuracy increase to 84%, but the network was still underfitting. 

* I finally used a finetuned ResNet 18 architecture to train the model with a learning rate of 0.0001 and an learning rate scheduler in case the loss does not reduce. 
```
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 3, verbose=True)
```

* I also added a L2 weight decay (value = 0.0001) to the adam optimizer to regularize the network. 

And voila, 25 epochs and a couple of episodes of FRIENDS later, the accuracy of the test set settled at 97%. 

## Strategies that I missed
* One strategy that would help a lot but I could not implement due to some time constraints is called as cutout. It is based on this paper: https://arxiv.org/abs/1708.04896
It randomly selects a rectangular area from the image and erases it's pixels. This helps the network learn not only the obvious features but also other features that it might miss. 

* Here is another one that I missed, it is called as mixup. It is essentially a convex combination of two images that is fed into the network. Now you might be wondering what the label would look like, it will be just the same. Here is the paper: https://arxiv.org/pdf/1710.09412.pdf

* I also did not use the entire dataset which has around 500,000 images due to some time constranits. If you can utilize these techiques to improve the accuracy further, please do not hesitate to initiate a discussion with me :)

## Usage

For now I have uploaded a model that contains the weights and you can use the below piece of code to load the weights.

```
model = torchvision.models.resnet18(pretrained=True)

model.fc = nn.Linear(512, 10)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Dropout(0.25),
    nn.Linear(512, 10),
    nn.LogSoftmax()
)

model.load_state_dict(torch.load("model_reg.pth"))
```

I will update the repo to include some command line arguments as soon as possible. 