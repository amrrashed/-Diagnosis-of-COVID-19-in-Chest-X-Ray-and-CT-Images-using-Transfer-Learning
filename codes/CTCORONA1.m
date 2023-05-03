clc;close all;clear;
%change augmenter%load images
digitDatasetPath = fullfile('D:\covid project\ADATASETS\Data_CT - mod\Data_CT');
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
    
    % Replacing the last layers with new layers
numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',2,'BiasLearnRateFactor',2);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);
%Train Network
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu',...
    'MiniBatchSize',9, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');
trainedNet = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(trainedNet,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)
%accuracy = mean(YPred == imdsValidation.Labels)


%%save Network
save covidnetimages2classCT.mat trainedNet lgraph

%for parallel processing
%p=gcp;
%delete(p)
%%Performance Study
figure;
plotconfusion(YValidation,YPred)
title('Confusion Matrix: ResNet');

% ROC Curve - Our target class is the first class in this scenario 
[fp_rate,tp_rate,T,AUC]=perfcurve(double(nominal(YValidation)),probs(:,1),1);
figure;
plot(fp_rate,tp_rate,'b-');
grid on;
xlabel('False Positive Rate');
ylabel('Detection Rate');
% Area under the ROC curve value
AUC
%evaluation
%Evaluate(YValidation,YPred)
ACTUAL=YValidation;
PREDICTED=YPred;
idx = (ACTUAL()=='covidct');
disp(idx)
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;

tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);

disp(['accuracy=' num2str(accuracy)])
disp(['sensitivity=' num2str(sensitivity)])
disp(['specificity=' num2str(specificity)])
disp(['precision=' num2str(precision)])
disp(['recall=' num2str(recall)])
disp(['f_measure=' num2str(f_measure)])
disp(['gmean=' num2str(gmean)])
    

