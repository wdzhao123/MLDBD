clear
clc
srcDir=uigetdir('/Users/weifei/Desktop/result_ctcug/SVD-jpg/'); 
cd(srcDir);
allnames=struct2cell(dir('*.jpg')); 

[k,len]=size(allnames);
if len==0
    fprintf('no pic\n');
else
    fprintf('there are %d pics\n',len);
end
for i=1:len
    image_name=allnames{1,i};
    im=imread(strcat('/Users/weifei/Desktop/result_ctcug/SVD-jpg/',image_name));
    [path,name,ext ] = fileparts(image_name);
    image_name = strcat( name,'.bmp');
    im = im2double(im);
    im = imresize(im, [320, 320]);
    %im = 1 - im;
    Path = fullfile('/Users/weifei/Desktop/result_ctcug/SVD/');
    imwrite(im,strcat(Path,image_name));
end