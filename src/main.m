% TIP Oximeter: A Deep Learning Framework for SpO2 Estimation from PPG Signals
% Author: Lei Zhao
% Date: 2025-02-18
% Description: This script trains and evaluates a deep learning model for SpO2 estimation
%              from PPG signals with optional pressure data integration.

%% Configuration Parameters
% Execution flags
trainNetworkFlag = 0;        % Set to 1 to train a new network
usingPressureFlag = 1;       % Set to 1 to include pressure data in the model

% Dataset configuration
datasetFolder = "dataset_folder";
testDisplaySampleName = "test_sample_name.mat";

% Training hyperparameters
NumEpochs = 300;
miniBatchSize = 16;
learnRate = 0.01;
gradDecay = 0.9;
sqGradDecay = 0.95;
segmentLength = 1024;

%% Initialize Environment
if trainNetworkFlag
    clearvars -except trainNetworkFlag usingPressureFlag datasetFolder testDisplaySampleName NumEpochs miniBatchSize learnRate gradDecay sqGradDecay segmentLength;
    fprintf('=== Training Mode ===\n');
else
    clearvars -except trainNetworkFlag usingPressureFlag datasetFolder testDisplaySampleName segmentLength;
    fprintf('=== Evaluation Mode ===\n');
    
    % Load pre-trained network if not in workspace
    if ~exist("TIPNet", "var")
        [filename, filepath] = uigetfile('*.mat', 'Load TIPNet weights', './log/');
        assert(filename(1) ~= '0', "Loading TIPNet weights failed");
        structWeight = load(fullfile(filepath, filename));
        TIPNet = structWeight.TIPNet;
        assert(exist("TIPNet", "var"), "No variable 'TIPNet' found in the selected file");
        
        % Parse configuration from filename
        [~, baseFileName, ~] = fileparts(filename);
        fprintf('Loaded model: %s\n', baseFileName);
        
        % Extract pressure flag from filename
        targetWord = 'withPressure';
        targetIndex = strfind(baseFileName, targetWord);
        if(~isempty(targetIndex))
            startIndex = targetIndex(1) + length(targetWord);
            if(startIndex < length(baseFileName))
                if(baseFileName(startIndex) == 't' || baseFileName(startIndex) == '1')
                    usingPressureFlag = true;
                    fprintf('Using pressure data: Yes\n');
                elseif(baseFileName(startIndex) == 'f' || baseFileName(startIndex) == '0')
                    usingPressureFlag = false;
                    fprintf('Using pressure data: No\n');
                else
                    fprintf('Warning: No "t"/"1" or "f"/"0" after "withPressure"\n');
                end
            else
                fprintf('Warning: No letter following "withPressure" in the filename\n');
            end
        else
            fprintf('Warning: No "withPressure" flag found in filename\n');
        end
    else
        fprintf("Using existing 'TIPNet' variable from workspace\n");
    end
end

%% Training Process
if trainNetworkFlag
    % Load training data
    fprintf('Loading training data...\n');
    trainDatasetFolder = fullfile(datasetFolder, "train");
    trainDS = signalDatastore(trainDatasetFolder, IncludeSubfolders=true, ...
        SignalVariableNames=["ppg_ir" "ppg_red" "pressure1" "pressure2" "spo2"]);
    trainDS = transform(trainDS, @(d,f,g)getInputSegments(d, segmentLength, usingPressureFlag));
    
    % Create network
    fprintf('Creating TIPNet architecture...\n');
    filterSize_short = 4;
    numFilters_short = 16;
    filterSize_long = 35;
    numFilters_long = 16;
    
    TIPNet = createTIPNet(usingPressureFlag, segmentLength, filterSize_short, numFilters_short, filterSize_long, numFilters_long);
    
    % Initialize training parameters
    iteration = 0;
    trainLossByIteration = [];
    
    % Create minibatch queue for training data
    mbq = minibatchqueue(trainDS, ...
        MiniBatchSize=miniBatchSize, ...
        MiniBatchFcn=@processMB, ...
        MiniBatchFormat=["CBT" "CBT"]);
    
    % Initialize Adam optimizer
    trailingAvg = [];
    trailingAvgSq = [];
    
    % Training loop
    fprintf('\n=== Starting Training ===\n');
    fprintf('Epochs: %d, Batch Size: %d, Learning Rate: %.4f\n', NumEpochs, miniBatchSize, learnRate);
    
    % Create a table for displaying progress
    fprintf('\n%-6s %-15s\n', 'Epoch', 'Training Loss');
    fprintf('%-6s %-15s\n', '-----', '-------------');
    
    for epoch = 1:NumEpochs
        % Reset and shuffle datastore for each epoch
        reset(mbq);
        shuffle(mbq);
        
        % Process each mini-batch
        while hasdata(mbq)
            iteration = iteration + 1;
            
            % Get next mini-batch
            [inputDataBatch, spo2Batch] = next(mbqTrain);
            
            % Evaluate model gradients and loss
            [trainloss, gradients, state] = dlfeval(@modelLoss, TIPNet, inputDataBatch, spo2Batch);
            trainLossByIteration(iteration) = trainloss;
            trainSumLoss = trainSumLoss + trainloss;
            
            % Update network state
            TIPNet.State = state;
            
            % Update network parameters using Adam optimizer
            [TIPNet, trailingAvg, trailingAvgSq] = adamupdate(TIPNet, gradients, ...
                trailingAvg, trailingAvgSq, iteration, learnRate, gradDecay, sqGradDecay); 
        end
        
        % Display progress
        fprintf('%-6d %-15.6f\n', epoch, double(trainLossByIteration(end)));
        
        % Save model periodically
        if mod(epoch, 10) == 0 || epoch == NumEpochs
            fprintf('Saving model at epoch %d...\n', epoch);
            latestModelPath = "log/TIPNet_withPressure" + usingPressureFlag + ...
                "_epochNum" + NumEpochs + "_mbSize" + miniBatchSize + "_learningRate" + learnRate + ...
                "_gradDecay" + gradDecay + "_sqGradDecay" + sqGradDecay + "_" + datasetFolder + ".mat";
            save(latestModelPath, "TIPNet");
        end
    end
    
    fprintf('\nTraining complete.\n');
    
    % Plot training progress
    figure('Name', 'Training Progress', 'NumberTitle', 'off');
    plot(1:iteration, mag2db(trainLossByIteration));
    grid on;
    title('Training Loss by Iteration');
    xlabel('Iteration');
    ylabel('Loss (dB)');
    axis tight;
end

%% Evaluation on Test Data
fprintf('\n=== Evaluating Model on Test Data ===\n');

% Load test data
testDatasetFolder = fullfile(datasetFolder, "test");
testDS = signalDatastore(testDatasetFolder, IncludeSubfolders=true, ...
    SignalVariableNames=["ppg_ir" "ppg_red" "pressure1" "pressure2" "spo2"]);
testDS = transform(testDS, @(d,f,g)getInputSegments(d, segmentLength, usingPressureFlag));

% Select specific test sample
idx = contains(string(testDS.UnderlyingDatastores{1}.Files), fullfile(testDisplaySampleName));
ds = subset(testDS, idx);
data = read(ds);
[inputBatch, spo2Batch] = processMB(data(:,1), data(:,2));

% Generate predictions
fprintf('Generating SpO2 predictions...\n');
inputData = dlarray(inputBatch, "CBT");
[dlpred_spo2] = predict(TIPNet, inputData);
pred_spo2 = squeeze(extractdata(dlpred_spo2))';
raw_spo2 = squeeze(spo2Batch)';
pred_spo2 = pred_spo2(:);

% Post-process predictions
pred_spo2_smooth = smoothdata(pred_spo2, 'movmedian', 10000);
pred_spo2_smooth = smoothdata(pred_spo2_smooth, 'movmean', 3000);
pred_spo2_smooth(pred_spo2_smooth > 100) = 100;

% Post-process ground truth
raw_spo2 = raw_spo2(:);
raw_spo2_smooth = smoothdata(raw_spo2, 'movmedian', 10000);
raw_spo2_smooth = smoothdata(raw_spo2_smooth, 'movmean', 3000);

% Calculate error metrics
mae_value = mae(pred_spo2_smooth - raw_spo2);
rmse_value = rms(pred_spo2_smooth - raw_spo2);
fprintf('Performance Metrics:\n');
fprintf('  Mean Absolute Error (MAE): %.4f\n', mae_value);
fprintf('  Root Mean Square Error (RMSE): %.4f\n', rmse_value);

% Visualize results
figure('Name', 'SpO2 Estimation Results', 'NumberTitle', 'off');
plot(raw_spo2_smooth, 'LineWidth', 1.5);
hold on;
plot(pred_spo2_smooth, 'LineWidth', 1.5);
grid on;
legend('Ground Truth', 'Predicted', 'Location', 'best');
title('SpO2 Estimation Results');
xlabel('Sample');
ylabel('SpO2 (%)');
ylim([85, 100]);

fprintf('\n=== Evaluation Complete ===\n');
