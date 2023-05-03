clc;
close all;
clear;

%for parallel processing
%delete(gcp('noncreate'))
%parpool

%alex net,resnet,darknet importer,google net,cnn,tensorflow and keras models
%deepNetworkDesigner

%load images
digitDatasetPath = fullfile('C:\Users\amr rashed\Desktop\Matlab\corona virus\moddataset2');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%load pretrained model
net = resnet18;

%analyzeNetwork(net) (replace final layers)
numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',2,'BiasLearnRateFactor',2);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

%Input resizing
%inputSize = net.Layers(1).InputSize;
% inputSize= [224 224];
% augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain);
% augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);

%Train Network
options = trainingOptions('sgdm', ...
    'MiniBatchSize',5, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');
trainedNet = trainNetwork(imdsTrain,lgraph,options);

[YPred,probs] = classify(trainedNet,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)%0.9867
%accuracy = mean(YPred == imdsValidation.Labels)

%%save Network
%save simpleDL.mat trainedNet lgraph

%for parallel processing
%p=gcp;
%delete(p)

 %% Try to classify something else
%img = readimage(imds,100);
%actualLabel = imds.Labels(100);
%predictedLabel = trainedNet.classify(img);
%imshow(img);
%title(['Predicted: ' char(predictedLabel) ', Actual: ' char(actualLabel)])