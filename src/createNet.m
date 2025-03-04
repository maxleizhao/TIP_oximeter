function lgraph = createNet(lgraph, filterSize, numFilters, branchStr)
% createNet - Creates a branch of the TIPNet architecture
%
% Inputs:
%   lgraph - Layer graph to add the network branch to
%   filterSize - Size of convolutional filters
%   numFilters - Base number of filters (will be scaled at deeper layers)
%   branchStr - Branch identifier ("short" for short-term or "long" for long-term)
%
% Output:
%   lgraph - Updated layer graph with the added branch

% Layer name conventions:
% - short/long indicates which branch the layer belongs to
% - ds means down sample (encoder), us means upsample (decoder)
% - i_j means ith row, jth layer

% Determine filter scaling based on branch type
numFiltScale = 1 + double(branchStr == "long");

% Set output layer name based on branch
if branchStr == "short"
    branchStrOutput = "outputLayer_short_targetSignal";
else
    branchStrOutput = "outputLayer_long_secondarySignal";
end

% Create long-term branch (smaller network)
if branchStr == "long"
    net = [
    % Row 1 encoder branch
    convolution1dLayer(filterSize, numFilters*4, Padding="same", Name="conv1d_"+branchStr+"_ds_1_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_1_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_1_1")
    
    convolution1dLayer(filterSize, numFilters*4, Padding="same", Name="conv1d_"+branchStr+"_ds_1_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_1_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_1_2")
    
    convolution1dLayer(filterSize, numFilters*4, Padding="same",Name="conv1d_"+branchStr+"_ds_1_3")
    batchNormalizationLayer("Name","batchnorm_"+branchStr+"_ds_1_3")
    leakyReluLayer(0.01,"Name","leakyrelu_"+branchStr+"_ds_1_3")
    
    averagePooling1dLayer(8, Padding="same", Stride=8, Name="avgpool1d_"+branchStr+"_1_to_2")

    convolution1dLayer(filterSize, numFilters*16*numFiltScale, Padding="same", Name="conv1d_"+branchStr+"_ds_2_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_3_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_3_1")
    
    convolution1dLayer(filterSize, numFilters*16*numFiltScale, Padding="same", Name="conv1d_"+branchStr+"_ds_2_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_3_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_3_2")
    
    averagePooling1dLayer(4, Padding="same", Stride=4, Name="avgpool1d_"+branchStr+"_2_to_3")
    ];

    lgraph = addLayers(lgraph, net);
end

% Create short-term branch (full U-Net architecture)
if branchStr == "short"
    net = [
    % Row 1 encoder branch
    convolution1dLayer(filterSize, numFilters, Padding="same", Name="conv1d_"+branchStr+"_ds_1_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_1_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_1_1")
    
    convolution1dLayer(filterSize, numFilters, Padding="same", Name="conv1d_"+branchStr+"_ds_1_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_1_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_1_2")
    
    convolution1dLayer(filterSize, numFilters, Padding="same",Name="conv1d_"+branchStr+"_ds_1_3")
    batchNormalizationLayer("Name","batchnorm_"+branchStr+"_ds_1_3")
    leakyReluLayer(0.01,"Name","leakyrelu_"+branchStr+"_ds_1_3")
    
    averagePooling1dLayer(2, Padding="same", Stride=2, Name="avgpool1d_"+branchStr+"_1_to_2")
    ];
    
    % Row 2 encoder branch
    net = [net
    convolution1dLayer(filterSize, numFilters*2, Padding="same", Name="conv1d_"+branchStr+"_ds_2_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_2_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_2_1")
    
    convolution1dLayer(filterSize, numFilters*2, Padding="same", Name="conv1d_"+branchStr+"_ds_2_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_2_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_2_2")
    
    averagePooling1dLayer(2, Padding="same", Stride=2, Name="avgpool1d_"+branchStr+"_2_to_3")
    ];
    
    % Row 3 encoder branch
    net = [net
    convolution1dLayer(filterSize, numFilters*4, Padding="same", Name="conv1d_"+branchStr+"_ds_3_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_3_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_3_1")
    
    convolution1dLayer(filterSize, numFilters*4, Padding="same", Name="conv1d_"+branchStr+"_ds_3_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_3_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_3_2")
    
    averagePooling1dLayer(2, Padding="same", Stride=2, Name="avgpool1d_"+branchStr+"_3_to_4")];
    
    % Add concatenation layer for long-term branch connection
    net = [net
        concatenationLayer(1, 2, Name="concat_"+branchStr+"_long_1")
        ];

    % Row 4 encoder branch
    net = [net
    convolution1dLayer(filterSize, numFilters*8, Padding="same", Name="conv1d_"+branchStr+"_ds_4_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_4_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_4_1")
    
    convolution1dLayer(filterSize, numFilters*8, Padding="same", Name="conv1d_"+branchStr+"_ds_4_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_4_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_4_2")
    
    averagePooling1dLayer(2, Padding="same", Stride=2, Name="avgpool1d_"+branchStr+"_4_to_5")
    ];
    
    % Row 5 encoder branch
    net = [net
    convolution1dLayer(filterSize, numFilters*16, Padding="same", Name="conv1d_"+branchStr+"_ds_5_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_5_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_5_1")
    
    convolution1dLayer(filterSize, numFilters*16, Padding="same", Name="conv1d_"+branchStr+"_ds_5_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_ds_5_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_ds_5_2")
    
    averagePooling1dLayer(2, Padding="same", Stride=2, Name="avgpool1d_"+branchStr+"_5_to_6")
    ];

    % Add concatenation layer for long-term branch connection
    net = [net
        concatenationLayer(1, 2, Name="concat_"+branchStr+"_long_2")
        ];
    
    % Row 6 - bridge
    net = [net
    convolution1dLayer(filterSize, numFilters*16*numFiltScale, Padding="same", Name="conv1d_"+branchStr+"_bridge_6_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_bridge_6_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_bridge_6_1")
    
    convolution1dLayer(filterSize, numFilters*16*numFiltScale, Padding="same", Name="conv1d_"+branchStr+"_bridge_6_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_bridge_6_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_bridge_6_2")
    
    transposedConv1dLayer(filterSize, numFilters*16*numFiltScale, Stride=2, Cropping="same", Name="transconv1d_"+branchStr+"_us_6_to_5")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_6_to_5")
    
    % Row 5 decoder branch
    concatenationLayer(1, 2, Name="concat_"+branchStr+"_5")
    
    convolution1dLayer(filterSize, numFilters*16, Padding="same", Name="conv1d_"+branchStr+"_us_5_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_5_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_5_1")
    
    convolution1dLayer(filterSize, numFilters*16, Padding="same", Name="conv1d_"+branchStr+"_us_5_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_5_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_5_2")
    
    transposedConv1dLayer(filterSize, numFilters*16, Stride=2, Cropping="same", Name="transconv1d_"+branchStr+"_us_5_to_4")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_5_to_4")
    
    % Row 4 decoder branch
    concatenationLayer(1, 2, Name="concat_"+branchStr+"_4")
    
    convolution1dLayer(filterSize, numFilters*8, Padding="same", Name="conv1d_"+branchStr+"_us_4_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_4_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_4_1")
    
    convolution1dLayer(filterSize, numFilters*8, Padding="same", Name="conv1d_"+branchStr+"_us_4_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_4_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_4_2")
    
    transposedConv1dLayer(filterSize, numFilters*8, Stride=2, Cropping="same", Name="transconv1d_"+branchStr+"_us_4_to_3")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_4_to_3")
    
    % Row 3 decoder branch
    concatenationLayer(1, 2, Name="concat_"+branchStr+"_3")
    
    convolution1dLayer(filterSize, numFilters*4, Padding="same", Name="conv1d_"+branchStr+"_us_3_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_3_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_3_1")
    
    convolution1dLayer(filterSize, numFilters*4, Padding="same", Name="conv1d_"+branchStr+"_us_3_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_3_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_3_2")
    
    transposedConv1dLayer(filterSize, numFilters*4, Stride=2, Cropping="same", Name="transconv1d_"+branchStr+"_us_3_to_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_3_to_2")
    
    % Row 2 decoder branch
    concatenationLayer(1, 2, Name="concat_"+branchStr+"_2")
    
    convolution1dLayer(filterSize, numFilters*2, Padding="same", Name="conv1d_"+branchStr+"_us_2_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_2_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_2_1")
    
    convolution1dLayer(filterSize, numFilters*2, Padding="same", Name="conv1d_"+branchStr+"_us_2_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_2_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_2_2")
    
    transposedConv1dLayer(filterSize, numFilters*2, Stride=2, Cropping="same", Name="transconv1d_"+branchStr+"_us_2_to_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_2_to_1")
    
    % Row 1 decoder branch
    concatenationLayer(1, 2, Name="concat_"+branchStr+"_1")
    
    convolution1dLayer(filterSize, numFilters, Padding="same", Name="conv1d_"+branchStr+"_us_1_1")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_1_1")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_1_1")
    
    convolution1dLayer(filterSize, numFilters, Padding="same", Name="conv1d_"+branchStr+"_us_1_2")
    batchNormalizationLayer(Name="batchnorm_"+branchStr+"_us_1_2")
    leakyReluLayer(0.01,Name="leakyrelu_"+branchStr+"_us_1_2")
    
    convolution1dLayer(filterSize, numFilters, Padding="same",Name="conv1d_"+branchStr+"_us_1_3")
    batchNormalizationLayer("Name","batchnorm_"+branchStr+"_us_1_3")
    leakyReluLayer(0.01,"Name","leakyrelu_"+branchStr+"_us_1_3")
    
    convolution1dLayer(filterSize, 1, Padding="same",Name=branchStrOutput)
    ];
    
    % Add layers to graph
    lgraph = addLayers(lgraph, net);
    
    % Connect skip connections within the short-term branch
    lgraph = connectLayers(lgraph, "leakyrelu_"+branchStr+"_ds_5_2", "concat_"+branchStr+"_5/in2");
    lgraph = connectLayers(lgraph, "leakyrelu_"+branchStr+"_ds_4_2", "concat_"+branchStr+"_4/in2");
    lgraph = connectLayers(lgraph, "leakyrelu_"+branchStr+"_ds_3_2", "concat_"+branchStr+"_3/in2");
    lgraph = connectLayers(lgraph, "leakyrelu_"+branchStr+"_ds_2_2", "concat_"+branchStr+"_2/in2");
    lgraph = connectLayers(lgraph, "leakyrelu_"+branchStr+"_ds_1_3", "concat_"+branchStr+"_1/in2");
end
end
