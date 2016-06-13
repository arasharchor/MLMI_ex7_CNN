clear all
close all
clc

% setup MatConvNet
run  /Users/Leonard/Documents/MATLAB/TUM/matconvnet/matlab/vl_setupnn

% load the ResNet-50 CNN
net = dagnn.DagNN.loadobj(load('imagenet-resnet-50-dag.mat')) ;
net.mode = 'test' ;

% load and preprocess an image
im = imread('peppers.png');
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

% run the CNN
net.eval({'data', im_}) ;

% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

% show the classification results
[bestScore, bestIndex] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{bestIndex}, bestIndex, bestScore)) ;

% top n (n=5) calssification results
top_n = 5;
scores_copy = scores;
for j = 1:length(scores_copy)
   [a, Index(j)] = max(scores_copy);
   scores_copy(Index(j)) = -inf;
end
maximumValues = scores(Index);
for i = 1:top_n
    sprintf('%s (%d), score %.3f',...
    net.meta.classes.description{Index(i)}, Index(i), ...
    maximumValues(i))
end

