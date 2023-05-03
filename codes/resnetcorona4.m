clc;close all;clear;
%change augmenter
%load images
digitDatasetPath = fullfile('G:\covid project\ADATASETS\matlabdb80');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(imds);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
imageSize = [224 224 3];
augimdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);
augimdsValidation = augmentedImageDatastore(imageSize,imdsValidation,'DataAugmentation',augmenter);
%Preview the random transformations applied to the first eight images in the image datastore.
minibatch = preview(augimdsTrain);
imshow(imtile(minibatch.input));
%load pretrained model
net = resnet18;
%analyzeNetwork(net) (replace final layers)
numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net);

% New Learnable Layer
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
   %analyzeNetwork(net) (replace final layers)
numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',2,'BiasLearnRateFactor',2);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);
%Train Network
options = trainingOptions('sgdm', ...
    'MiniBatchSize',9, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');
trainedNet = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(trainedNet,augimdsValidation);
%YValidation = augimdsValidation.Labels;
%accuracy = sum(YPred == YValidation)/numel(YValidation)
%accuracy = mean(YPred == imdsValidation.Labels)

%%save Network
save covidnetimages2classCT.mat trainedNet lgraph

%for parallel processing
%p=gcp;
%delete(p)

