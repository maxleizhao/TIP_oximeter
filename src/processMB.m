function [inputDataBatch, spo2Batch] = processMB(inputDataCell, spo2Cell)
% processMB - Processes minibatches for TIPNet training and inference
%
% This function converts cell arrays of input and target data into concatenated 
% arrays suitable for network training and inference. It transforms the data
% from individual segments in cells to a batch format.
%
% Inputs:
%   inputDataCell - Cell array containing input signal segments
%                   Each cell contains one segment with format [C×1×T]
%                   where C is number of channels (2 or 4) and T is segment length
%   spo2Cell - Cell array containing target SpO2 values
%              Each cell contains one segment with format [1×1×T]
%
% Outputs:
%   inputDataBatch - Concatenated input data with format [C×B×T]
%                    where B is the batch size (number of segments)
%   spo2Batch - Concatenated target data with format [1×B×T]
%
% Note: This function is used by the minibatchqueue during training
% and for preparing test data during inference.

% Concatenate input data cells along the batch dimension (dim 2)
% This converts from a cell array of [C×1×T] arrays to a single [C×B×T] array
inputDataBatch = cat(2, inputDataCell{:});

% Concatenate target data cells along the batch dimension (dim 2)
% This converts from a cell array of [1×1×T] arrays to a single [1×B×T] array
spo2Batch = cat(2, spo2Cell{:});

end
