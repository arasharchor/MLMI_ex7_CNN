% setup MatConvNet
path = '/Users/Leonard/Google Drive/[TUM]/3. SoSe 2016/Machine Learning in Medical Imaging/07 CNN/ex07/matconvnet';
run fullfile(path, vl_setupnn.m) ;
% load the pre-trained ResNet 50 (dagnn)
net = dagnn.DagNN.loadobj(load('imagenet-resnet-50-dag.mat')) ;
net.mode = 'test' ;
% load and preprocess an image
im = imread('trumpet.jpg') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
% run the CNN
net.eval({'data', im_}) ;

%net.print({'data', [224 224 3]}, 'all', true)

% get the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;
[top, index] = sort(scores(:),'descend');
prob = [top(1);top(2);top(3);top(4);top(5)];
table(prob,'RowNames',{net.meta.classes.description{index(1)};net.meta.classes.description{index(2)};net.meta.classes.description{index(3)};net.meta.classes.description{index(4)};net.meta.classes.description{index(5)}})

class = net.meta.classes.description{index(1)};
classIdx = index(1);
classProb = top(1);
ausgang = zeros(16,16);
for i = 0:15
    for j = 0:15
        imOcc = im_;
        occlusion = ones(14,14,3);
        rgbMean = mean(mean(im));
        occlusion(:,:,1) = rgbMean(:,:,1);
        occlusion(:,:,2) = rgbMean(:,:,2);
        occlusion(:,:,3) = rgbMean(:,:,3);
        imOcc(i*14+1:i*14+14,j*14+1:j*14+14,:) = occlusion;
        net.eval({'data', imOcc}) ;
        scoresOcc = net.vars(net.getVarIndex('prob')).value ;
        scoresOcc = squeeze(gather(scoresOcc)) ;
        ausgang(i+1,j+1) = classProb - scoresOcc(classIdx);
    end
end

ausgang = ausgang - min(min(ausgang));
ausgang = 255*abs(ausgang)/max(max(abs(ausgang)));
ausgang = ind2rgb(int16(imresize(ausgang,14,'nearest')),colormap(jet(255)));

% show the classification results
[bestScore, best] = max(scores) ;
C = imfuse(im,ausgang,'blend');
figure(1) ; clf ; imshow(C) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;