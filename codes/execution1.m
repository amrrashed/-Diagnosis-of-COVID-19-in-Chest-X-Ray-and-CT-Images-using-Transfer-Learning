clear;clc;close
tic
% Access the trained model 
%JapaneseVowelsNet LSTM Network
%https://www.mathworks.com/help/deeplearning/ref/predict.html
%net = densenet201();%net = googlenet; %net = nasnetmobile();
%net = resnet50();%net = resnet18();
net = resnet50();
%net = alexnet;%net = vgg19();

% See details of the architecture 
net.Layers

% Read the image to classify 

%I = imread('mug.jpg');%vgg16_81%resnet50_95.27%
%I = imread('dog.jpg');%vgg16_98.2%resnet50_97.08%
I = imread('cat.jpg');%vgg16_98.12%resnet50_95.95%

% Adjust size of the image 
sz = net.Layers(1).InputSize ;
I=imresize(I,[sz(1) sz(2)]);
% Classify the image using DenseNet-201 
label = classify(net, I)
YPred = predict(net, I);
%[YPred,scores] = classify(net,I)
max(YPred)
 toc;%Elapsed time is 2.421189 seconds.
% Show the image and the classification results 
figure 
imshow(I) 
text(10,20,strcat(char(label),num2str(max(YPred)*100),'%'),'Color','red')