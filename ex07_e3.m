clc
close all
clear
    
%Create Max-Pooling Block
poolBlock = dagnn.Pooling('poolSize', [2 2], 'stride', 2);


% Setup MatConvNet
run  /Users/Leonard/Documents/MATLAB/TUM/matconvnet/matlab/vl_setupnn
% Create DAGNN object
net = dagnn.DagNN();


% Add Conv layer (x1 = in, x2 = out, paramerser: filter f0x, biases b0x)
net.addLayer('conv_3x3_01', convBlock(3,3,3,32), {'x01'}, {'x02'}, ...
                 {'f01', 'b01'});
% Add ReLU layer
net.addLayer('relu_01', dagnn.ReLU(), {'x02'}, {'x03'}, {});
% Add Max-Pooling layer
net.addLayer('mpool_01', poolBlock, {'x03'}, {'x04'}, {});
  
net.addLayer('conv_3x3_02', convBlock(3,3,32,64), {'x04'}, {'x05'}, ...
                 {'f02', 'b02'});
net.addLayer('relu_02', dagnn.ReLU(), {'x05'}, {'x06'}, {});
net.addLayer('mpool_02', poolBlock, {'x06'}, {'x07'}, {});

net.addLayer('conv_3x3_03', convBlock(3,3,64,128), {'x07'}, {'x08'}, ...
             {'f03', 'b03'});
net.addLayer('relu_03', dagnn.ReLU(), {'x08'}, {'x09'}, {});
net.addLayer('mpool_03', poolBlock, {'x09'}, {'x10'}, {});

net.addLayer('conv_3x3_04', convBlock(3,3,128,256), {'x10'}, {'x11'}, ...
             {'f04', 'b04'});
net.addLayer('relu_04', dagnn.ReLU(), {'x11'}, {'x12'}, {});
net.addLayer('mpool_04', poolBlock, {'x12'}, {'x13'}, {});

% Add fully-connected layer
net.addLayer('conv_fc_05', convBlock(3,3,256,2048), {'x13'},{'x14'},...
             {'f05', 'b05'});
net.addLayer('relu_05', dagnn.ReLU(), {'x14'}, {'x15'}, {});
    
net.addLayer('conv_fc_06', convBlock(1,1,2048,2), {'x15'},{'x16'},...
             {'f06', 'b06'});
net.addLayer('relu_06', dagnn.ReLU(), {'x16'}, {'x17'}, {});


%Initialise random parameters
net.initParams();

%Visualize Network
net.print({'x01', [64 64 3]})
