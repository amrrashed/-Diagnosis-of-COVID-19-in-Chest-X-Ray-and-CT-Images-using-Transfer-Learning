clc;close all;clear;
%%data augmentation rotation -15:15 and x axis reflection
%for parallel processing
%delete(gcp('noncreate'))
%parpool
%alex net,resnet,darknet importer,google net,cnn,tensorflow and keras models
%deepNetworkDesigner
%load images
digitDatasetPath = fullfile('D:\CIT project\datasets\ultrasonic images\us-dataset\originals');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
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
%save covidnet79images1.mat trainedNet lgraph

%Display four sample validation images with their predicted labels.
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%accuracy
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

%for parallel processing
%p=gcp;
%delete(p)

