function lgraph = createTIPNet(usingPressureFlag, inputSize, filterSize_short, numFilters_short, filterSize_long, numFilters_long)
% createTIPNet - Creates the TIP Oximeter neural network architecture
%
% Inputs:
%   usingPressureFlag - Boolean flag indicating whether to include pressure data
%   inputSize - Length of input sequence
%   filterSize_short - Filter size for the short-term branch
%   numFilters_short - Number of filters for the short-term branch
%   filterSize_long - Filter size for the long-term branch
%   numFilters_long - Number of filters for the long-term branch
%
% Output:
%   lgraph - Layer graph object representing the TIPNet architecture

% Initialize empty layer graph
lgraph = layerGraph;

% Create input layer based on whether pressure data is included
if usingPressureFlag
    inputLayer = sequenceInputLayer(4, MinLength=inputSize, Name="inputMixture");
else
    inputLayer = sequenceInputLayer(2, MinLength=inputSize, Name="inputMixture");
end
lgraph = addLayers(lgraph, inputLayer);

% Add short-term branch (previously main branch)
lgraph = createNet(lgraph, filterSize_short, numFilters_short, "short");
lgraph = connectLayers(lgraph, 'inputMixture', 'conv1d_short_ds_1_1');

% Add long-term branch (previously fast branch)
lgraph = createNet(lgraph, filterSize_long, numFilters_long, "long");
lgraph = connectLayers(lgraph, 'inputMixture', 'conv1d_long_ds_1_1');

% Connect long-term branch to short-term branch at appropriate points
lgraph = connectLayers(lgraph, "avgpool1d_long_1_to_2", "concat_short_long_1/in2");
lgraph = connectLayers(lgraph, "avgpool1d_long_2_to_3", "concat_short_long_2/in2");

end