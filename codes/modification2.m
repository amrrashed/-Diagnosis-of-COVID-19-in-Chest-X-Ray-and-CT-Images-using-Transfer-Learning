clc;clear
digitDatasetPath = fullfile('G:\new researches\COVID paper\codes\MATLAB CODES\ADATASETS\chest-xray-pneumonia\chest_xray\DB224enhanced');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labels=countEachLabel(imds);
img2=zeros(224,224,3);
parfor i=1:length(imds.Labels)
img=readimage(imds,i);
img1 = uint8(imresize(img,[224 224]));
c=length(size(img));
if c==2
img2=cat(3,img1,img1,img1);
imwrite(img2,cell2mat(imds.Files(i)))
elseif c==3
 imwrite(img1,cell2mat(imds.Files(i)))
end
%solve bitdepth error
%aa=imfinfo(imds.Files{perms(i)});
%B(i)=aa.BitDepth;
end

