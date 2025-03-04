function [loss, grads, state] = modelLoss(net, inputDataBatch, spo2Batch)
% modelLoss - Calculates loss and gradients for TIPNet training
%
% This function computes the loss between predicted and target SpO2 values
% and calculates gradients for network parameter updates during training.
%
% Inputs:
%   net - The TIPNet network object
%   inputDataBatch - Input data batch with format [C×B×T]
%                   where C is channels, B is batch size, T is time steps
%   spo2Batch - Target SpO2 values with format [1×B×T]
%
% Outputs:
%   loss - Scalar loss value (Mean Absolute Error)
%   grads - Gradients of the loss with respect to network parameters
%   state - Updated network state
%
% Note: Alternative loss functions are included as comments for reference

% Forward pass through the network
[spo2Pred, state] = net.forward(inputDataBatch);

% Calculate Mean Absolute Error (MAE) loss
loss = stripdims(mean(abs(spo2Batch - spo2Pred), "all"));

% Calculate gradients of the loss with respect to network parameters
grads = dlgradient(loss, net.Learnables);

% Convert loss to double for reporting
loss = double(gather(extractdata(loss)));

end