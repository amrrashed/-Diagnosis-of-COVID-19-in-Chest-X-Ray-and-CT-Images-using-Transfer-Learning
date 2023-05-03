%%F:\heart and lung\chest-xray-pneumonia\chest_xray\test
clc;clear
digitDatasetPath = fullfile('F:\heart and lung\chest-xray-pneumonia\chest_xray\train');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labels=countEachLabel(imds);
for i=1:length(imds.Labels)
img=readimage(imds,i);
[a,b,c(i)]=size(img);
end
k=find(c==3);
%mkdir 'C:\Users\amr rashed\Desktop\Matlab\corona virus\newnormal'
folderpath='C:\Users\amr rashed\Desktop\Matlab\corona virus\newnormal\';
for i=1:128
img2=readimage(imds,k(i));
imwrite(img2,strcat(folderpath,mat2str(i),'.jpeg'))
end
