function outputCell = getInputSegments(cellInput, segmentLength, usingPressureFlag)
% getInputSegments - Segments input signals into fixed-length windows for TIPNet
%
% Inputs:
%   cellInput - Cell array containing the input signals:
%               {1}: ppg_ir - Infrared PPG signal
%               {2}: ppg_red - Red PPG signal
%               {3}: pressure1 - First pressure signal
%               {4}: pressure2 - Second pressure signal
%               {5}: spo2 - Ground truth SpO2 values
%   segmentLength - Length of each segment (number of samples)
%   usingPressureFlag - Boolean flag indicating whether to include pressure data
%
% Output:
%   outputCell - Cell array containing segmented input and target data
%                Each row contains [inputData, targetData] for one segment

% Extract signals from cell input
ppg_ir = cellInput{1};
ppg_red = cellInput{2};
pressure1 = cellInput{3};
pressure2 = cellInput{4};
spo2 = cellInput{5};

% Segment the data using buffer function
% This creates segments of length segmentLength from the continuous signals
[idxs, ~] = buffer(1:size(ppg_ir,1), segmentLength);

% Convert to single precision and transpose to get segments in rows
ppg_ir = single(ppg_ir(idxs)');
ppg_red = single(ppg_red(idxs)');
pressure1 = single(pressure1(idxs)');
pressure2 = single(pressure2(idxs)');
spo2 = single(spo2(idxs)');

% Get number of segments
numSegments = size(ppg_ir, 1);

% Reshape to CBT format: C=channels, B=batch, T=time
% For input data: C=1 (per signal), B=numSegments, T=segmentLength
ppg_ir = reshape(ppg_ir, 1, numSegments, []);
ppg_red = reshape(ppg_red, 1, numSegments, []);
pressure1 = reshape(pressure1, 1, numSegments, []);
pressure2 = reshape(pressure2, 1, numSegments, []);

% Concatenate input channels based on pressure flag
if usingPressureFlag
    % Include all 4 channels: IR PPG, Red PPG, Pressure1, Pressure2
    inputData = cat(1, ppg_ir, ppg_red, pressure1, pressure2);
else
    % Include only PPG channels: IR PPG, Red PPG
    inputData = cat(1, ppg_ir, ppg_red);
end

% Reshape SpO2 data to match input format
spo2 = reshape(spo2, 1, numSegments, []);

% Convert matrices to cell arrays for datastore compatibility
% Each cell contains one segment of data
if usingPressureFlag
    % Create cells with 4 channels (C=4, B=1, T=segmentLength)
    inputDataCell = mat2cell(inputData, 4, ones(numSegments, 1), segmentLength)';
else
    % Create cells with 2 channels (C=2, B=1, T=segmentLength)
    inputDataCell = mat2cell(inputData, 2, ones(numSegments, 1), segmentLength)';
end

% Create target data cells (C=1, B=1, T=segmentLength)
spo2Cell = mat2cell(spo2, 1, ones(numSegments, 1), segmentLength)';

% Combine input and target data cells
outputCell = [inputDataCell spo2Cell];

end