clc;
clear;
close all;

digitDatasetPath = fullfile('G:\covid project\matlabdb80');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%outputSize = [224 224 3];
%auimds = augmentedImageDatastore(outputSize,imds);
%img2=zeros(224,224,3);
%for i=1:length(imds.Labels)
 %   img=imresize(readimage(imds,i), [224,224]);
  %  [a b c]=size(img);
   % imwrite(img,cell2mat(imds.Files(i)))
%end

for i=1:length(imds.Labels)
[img,map]=readimage(imds,i);
 %imshow(img)
a(:,:,:,i) = ind2rgb(img,map);
img1 = imresize(a,[224 224]);
%img2(:,:,1)=img1;
%img2(:,:,2)=img1;
%img2(:,:,3)=img1;
%img1= cat(3, img(:,:,1), img(:,:,2), img(:,:,3) )
 imwrite(img1,cell2mat(imds.Files(i)))
end
