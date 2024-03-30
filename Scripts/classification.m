% Letter Classification Code
% Testing Various Classifier types (Basic Neural Network, Euclidean, Bayesian, KNN, SVM)
close all;
clear;

% Load features data
trainFeatures = load_features("train");
validationFeatures = load_features("validation");
testFeatures = load_features("test");

%% All possible characters
% chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

% Convert features to matrix
normalize = 0;
[trainData, trainLabels, trainLetters, trainLabelMatrix] = ...
    convertFeaturesToMatrix(trainFeatures, chars, normalize);
[validationData, validationLabels, validationLetters, validationLabelMatrix] = ...
    convertFeaturesToMatrix(validationFeatures, chars, normalize);
[testData, testLabels, testLetters, testLabelMatrix] = ...
    convertFeaturesToMatrix(testFeatures, chars, normalize);

%% Deep Neural Network Classifier

% Define Network Architecture
numFeatures = size(trainData,2);
numClasses = numel(unique(trainLabels));

layers = [
    featureInputLayer(numFeatures,Normalization="zscore")
    fullyConnectedLayer(200)
    batchNormalizationLayer
    reluLayer
    % dropoutLayer(0.2)
    fullyConnectedLayer(100)
    batchNormalizationLayer
    reluLayer
    % dropoutLayer(0.05)
    fullyConnectedLayer(50)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.02)
    fullyConnectedLayer(numClasses)
    softmaxLayer];

% Set training options
Options = trainingOptions('adam',...
    MaxEpochs=50,...
    MiniBatchSize=4096, ...
    LearnRateSchedule='piecewise',...
    Metrics='accuracy',...
    ValidationPatience=5,...
    ... % ExecutionEnvironment='parallel-cpu',...
    Shuffle='every-epoch',...
    ValidationData={validationData, categorical(validationLetters)'}, ...
    ValidationFrequency=500, ...  % Validate every 500 iterations
    ... % Verbose=true,...  % Display training progress
    Plots="training-progress",...
    OutputNetwork = "best-validation-loss",...
    InitialLearnRate = 0.01...
);

% Train the network
dlNetwork = trainnet(trainData, categorical(trainLetters)', layers, 'crossentropy', Options);
% analyzeNetwork(netTrained)
% save('Results/Models/dlNetwork_7.mat','dlNetwork');

%% Test the deep neural network
% load('Results/Models/dlNetwork_6.mat');
fprintf('\nDeep Neural Network Classifier\n')

% Train
trainScores = predict(dlNetwork,trainData);
trainEst = arrayfun(@(i) find(trainScores(i,:)==max(trainScores(i,:))),1:size(trainScores,1));
trainError = sum(trainLabels ~= trainEst)/numel(trainLabels);
fprintf('     Training Error: %.4f \n', trainError)
% Validate
validScores = predict(dlNetwork,validationData);
validEst = arrayfun(@(i) find(validScores(i,:)==max(validScores(i,:))),1:size(validScores,1));
validError = sum(validationLabels ~= validEst)/numel(validationLabels);
fprintf('     Validation Error: %.4f \n', validError)
% Test
testScores = predict(dlNetwork,testData);
testEst = arrayfun(@(i) find(testScores(i,:)==max(testScores(i,:))),1:size(testScores,1));
testError = sum(testLabels ~= testEst)/numel(testLabels);
fprintf('     Test Error: %.4f \n', testError)

% Plot Confusion Matrices for Deep NN Classifier
toCaptialLetters = @(labels) categorical(labels,1:26,num2cell(chars(1:26)));

trainCat = toCaptialLetters(trainLabels);
valCat = toCaptialLetters(validationLabels);
testCat = toCaptialLetters(testLabels);
trainEstCat = toCaptialLetters(trainEst);
valEstCat = toCaptialLetters(validEst);
testEstCat = toCaptialLetters(testEst);
allLabs = [trainCat, valCat, testCat];
allEst = [trainEstCat, valEstCat, testEstCat];
allError = sum(allLabs ~= allEst)/numel(allLabs);

figure
sgtitle('Deep Neural Network Classifier Results');
trainTitle = ['Train (Accuracy: ', num2str(100*(1-trainError)),'%)'];
subplot(2,2,1); confusionchart(trainCat,trainEstCat,'Title',trainTitle,...
    'ColumnSummary','column-normalized','RowSummary','row-normalized');
validTitle = ['Validate (Accuracy: ', num2str(100*(1-validError)),'%)'];
subplot(2,2,2); confusionchart(valCat,valEstCat,'Title',validTitle,...
    'ColumnSummary','column-normalized','RowSummary','row-normalized');
testTitle = ['Test (Accuracy: ', num2str(100*(1-testError)),'%)'];
subplot(2,2,3); confusionchart(testCat,testEstCat,'Title',testTitle,...
    'ColumnSummary','column-normalized','RowSummary','row-normalized');
allTitle = ['Overall (Accuracy: ', num2str(100*(1-allError)),'%)'];
subplot(2,2,4); confusionchart(allLabs,allEst,'Title',allTitle,...
    'ColumnSummary','column-normalized','RowSummary','row-normalized');

%% Train basic (shallow) neural network
% Generated using nprtool
x = [trainData' validationData' testData'];
t = [trainLabelMatrix' validationLabelMatrix' testLabelMatrix'];

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = [10 5];
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
nTrain = size(trainData,1);
nVal = size(validationData,1);
nTest = size(testData,1);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:nTrain;
net.divideParam.valInd = nTrain+1:nTrain+nVal;
net.divideParam.testInd= nTrain+nVal+1:nTrain+nVal+nTest;

% Train the Network
[net,tr] = train(net,x,t);

% Save the trained network
basicNetworkResults.Network = net;
basicNetworkResults.TrainingResults = tr;
save('Results/Models/basicNetworkResults_2.mat','basicNetworkResults');

%% Test the basic network
load('Results/Models/basicNetworkResults_2.mat'); % Load trained network
basicNet = basicNetworkResults.Network;
tr = basicNetworkResults.TrainingResults;
fprintf('\nBasic Neural Network Classifier\n')

% Uncomment to View the Network
% view(basicNet)

% Train
trainEst = basicNet(trainData');
trainLabsInd = vec2ind(trainLabelMatrix');
trainEstInd = vec2ind(trainEst);
trainError = sum(trainLabsInd ~= trainEstInd)/numel(trainLabsInd);
fprintf('     Training Error: %.4f \n', trainError)
e = gsubtract(trainLabelMatrix',trainEst);
performance = perform(basicNet,trainLabelMatrix',trainEst); % Mean-squared error
% Uncomment these lines to enable various plots for training performance.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, ploterrhist(e)
% figure, plotconfusion(trainLabelMatrix',trainEst)
% figure, plotroc(trainLabelMatrix',trainEst)

% Validate
validEst = basicNet(validationData');
validLabsInd = vec2ind(validationLabelMatrix');
validEstInd = vec2ind(validEst);
validError = sum(validLabsInd ~= validEstInd)/numel(validLabsInd);
fprintf('     Validation Error: %.4f \n', validError)
% Test
testEst = basicNet(testData');
testLabsInd = vec2ind(testLabelMatrix');
testEstInd = vec2ind(testEst);
testError = sum(testLabsInd ~= testEstInd)/numel(testLabsInd);
fprintf('     Test Error: %.4f \n', testError)

% Plot Confusion Matrices for Basic NN Classifier
toCaptialLetters = @(labels) categorical(labels,1:26,num2cell(chars(1:26)));

trainCat = toCaptialLetters(trainLabsInd);
valCat = toCaptialLetters(validLabsInd);
testCat = toCaptialLetters(testLabsInd);
trainEstCat = toCaptialLetters(trainEstInd);
valEstCat = toCaptialLetters(validEstInd);
testEstCat = toCaptialLetters(testEstInd);
allLabs = [trainCat, valCat, testCat];
allEst = [trainEstCat, valEstCat, testEstCat];

figure
sgtitle('Basic Neural Network Classifier Results');
subplot(2,2,1); confusionchart(trainCat,trainEstCat,'Title','Train',...
    'ColumnSummary','column-normalized','RowSummary','row-normalized');
subplot(2,2,2); confusionchart(valCat,valEstCat,'Title','Validate',...
    'ColumnSummary','column-normalized','RowSummary','row-normalized');
subplot(2,2,3); confusionchart(testCat,testEstCat,'Title','Test',...
    'ColumnSummary','column-normalized','RowSummary','row-normalized');
subplot(2,2,4); confusionchart(allLabs,allEst,'Title','Overall',...
    'ColumnSummary','column-normalized','RowSummary','row-normalized');

%% Calculate statistics for Euclidean & Bayesian classifiers
[N, numFeatures] = size(trainData);
numClasses = length(chars);
means = zeros([numFeatures,numClasses]);
covs = zeros([numFeatures,numFeatures,numClasses]);
P = zeros([1, numClasses]);
for c=1:numClasses
    classData = trainData(trainLabels==c,:);
    classMean = mean(classData);
    means(:,c) = classMean';
    covar = cov(classData);
    if anynan(covar)
        covar = eye(numFeatures)*1e-10;
    end
    covs(:,:,c) = covar;
    P(c) = length(classData)/length(trainData);
end

% Euclidean Classifier
fprintf('\nEuclidean Classifier\n')
% Train
z=euclidean_classifier(means,trainData');
euclideanError = sum(trainLabels~=z)/length(z);
fprintf('     Training Error: %.4f \n', euclideanError)
% Validate
z=euclidean_classifier(means,validationData');
euclideanError = sum(validationLabels~=z)/length(z);
fprintf('     Validation Error: %.4f \n', euclideanError)
% Test
z=euclidean_classifier(means,testData');
euclideanError = sum(testLabels~=z)/length(z);
fprintf('     Test Error: %.4f \n', euclideanError)

% Bayesian Classifier
fprintf('\nBayesian Classifier\n')
% Train
z = bayes_classifier(means,covs,P,trainData');
error = sum(trainLabels~=z)/length(z);
fprintf('     Training Error: %.4f \n', error)
% Validate
z = bayes_classifier(means,covs,P,validationData');
error = sum(validationLabels~=z)/length(z);
fprintf('     Validation Error: %.4f \n', error)
% Test
z = bayes_classifier(means,covs,P,testData');
error = sum(testLabels~=z)/length(z);
fprintf('     Test Error: %.4f \n', error)

% Naive Bayes Classifier
fprintf('\nNaive Bayes Classifier\n')
% Train
Mdl = fitcnb(trainData,trainLabels);
z = predict(Mdl,trainData);
error = sum(trainLabels~=z')/length(z);
fprintf('     Training Error: %.4f \n', error)
% Validate
z = predict(Mdl,validationData);
error = sum(validationLabels~=z')/length(z);
fprintf('     Validation Error: %.4f \n', error)
% Test
z = predict(Mdl,testData);
error = sum(testLabels~=z')/length(z);
fprintf('     Test Error: %.4f \n', error)

% KNN Classifier
fprintf('\nKNN Classifier\n')
% Train
Mdl = fitcknn(trainData,trainLabels,'NumNeighbors',10);
zTrain = predict(Mdl,trainData);
error = sum(trainLabels~=zTrain')/length(zTrain);
fprintf('     Training Error: %.4f \n', error)
% Validate
zVal = predict(Mdl,validationData);
error = sum(validationLabels~=zVal')/length(zVal);
fprintf('     Validation Error: %.4f \n', error)
% Test
zTest = predict(Mdl,testData);
error = sum(testLabels~=zTest')/length(zTest);
fprintf('     Test Error: %.4f \n', error)

% Plot Confusion Matrices for KNN Classifier
toCaptialLetters = @(labels) categorical(labels,1:26,num2cell(chars(1:26)));

trainLabelMatrix = toCaptialLetters(trainLabels);
valLabs = toCaptialLetters(validationLabels);
testLabs = toCaptialLetters(testLabels);
zzTrain = toCaptialLetters(zTrain);
zzVal = toCaptialLetters(zVal);
zzTest = toCaptialLetters(zTest);
allLabs = [trainLabelMatrix, valLabs, testLabs];
zzAll = [zzTrain; zzVal; zzTest;];

figure
sgtitle('KNN Classifier Results');
subplot(2,2,1); confusionchart(trainLabelMatrix,zzTrain,'Title','Train');
subplot(2,2,2); confusionchart(valLabs,zzVal,'Title','Validate');
subplot(2,2,3); confusionchart(testLabs,zzTest,'Title','Test');
subplot(2,2,4); confusionchart(allLabs,zzAll,'Title','Overall');
