clc;clear all;close all
%x=cell(4,5);
path='G:\new researches\COVID paper\codes\MATLAB CODES\ADATASETS\COVID-19 Dataset-Mendely-224\COVID-19 Dataset\X-ray-size 224';
[t,x]=efficientnetb0cv(path,'adam',1,5);
t=ceil(t.*10000)/10000;
pause(60)
[t1,x1]=efficientnetb0cv(path,'rmsprop',1,5);
t1=ceil(t1.*10000)/10000;
pause(60)
[t2,x2]=efficientnetb0cv(path,'sgdm',1,5);
t2=ceil(t2.*10000)/10000;
pause(60)
title={'''AUC','''accuracy','''sensitivity','''specificity','''precision','''recall','''f_measure','''gmean'};
total=[t;t1;t2];
filename='performance.xlsx';
% xlswrite(filename,title,'Sheet1','A1')
% xlswrite(filename,total,'Sheet1','A2')
writecell(title,filename,'Sheet',2,'Range','A1:H1')
writematrix(total,filename,'Sheet',2,'Range','A2:H4')
winopen(filename);
