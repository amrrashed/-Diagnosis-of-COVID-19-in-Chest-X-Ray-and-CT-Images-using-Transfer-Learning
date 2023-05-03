clear;clc;close
%JapaneseVowelsNet LSTM Network
%https://www.mathworks.com/help/deeplearning/ref/predict.html
% Access the trained model 
%net = densenet201();%net = googlenet; 
%net = nasnetmobile();%net = resnet50();
%net = resnet18();%net = vgg16();%net = alexnet;
net =densenet201();

% See details of the architecture 
net.Layers

% Read the image to classify 
I = imread('peppers.png');

% Adjust size of the image 
sz = net.Layers(1).InputSize 
I = I(1:sz(1),1:sz(2),1:sz(3));

% Classify the image using DenseNet-201 
label = classify(net, I)

% Show the image and the classification results 
figure 
imshow(I) 
text(10,20,char(label),'Color','white')