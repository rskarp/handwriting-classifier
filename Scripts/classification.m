% Letter Classification Code
% Testing Various Classifier types (Euclidean, Bayesian, KNN, SVM)
close all;
clear;

% Load features data
trainFeatures = load('Features/train/train_features_1.mat').data;
validationFeatures = load('Features/validation/validation_features_1.mat').data;
testFeatures = load('Features/test/test_features_1.mat').data;

%% All possible characters
chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

% Convert features to matrix
[trainData, trainLabels] = convertFeaturesToMatrix(trainFeatures, chars);
[validationData, validationLabels] = convertFeaturesToMatrix(validationFeatures, chars);
[testData, testLabels] = convertFeaturesToMatrix(testFeatures, chars);

% Normalize feature columns
trainData = normc(trainData);
validationData = normc(validationData);
testData = normc(testData);

% Calculate statistics for Euclidean & Bayesian classifiers
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

trainLabs = toCaptialLetters(trainLabels);
valLabs = toCaptialLetters(validationLabels);
testLabs = toCaptialLetters(testLabels);
zzTrain = toCaptialLetters(zTrain);
zzVal = toCaptialLetters(zVal);
zzTest = toCaptialLetters(zTest);
allLabs = [trainLabs, valLabs, testLabs];
zzAll = [zzTrain; zzVal; zzTest;];

figure
sgtitle('KNN Classifier Results');
subplot(2,2,1); confusionchart(trainLabs,zzTrain,'Title','Train');
subplot(2,2,2); confusionchart(valLabs,zzVal,'Title','Validate');
subplot(2,2,3); confusionchart(testLabs,zzVal,'Title','Test');
subplot(2,2,4); confusionchart(allLabs,zzAll,'Title','Overall');

%% Test Unknown data using trained classifier
filenames = unique(cellfun(@(x) x.Filename, trainFeatures));
% Generate text string for each input data file
for i = 1:length(filenames)
    fname = filenames(i);
    % Get all letters in the file
    letters = trainFeatures(cellfun(@(x) x.Filename==fname, trainFeatures));
    for l = 1:length(letters)
        % TODO: Classify each letter
    end
end

%% Function Definitions

% Convert numeric feature data to matrix & get class labels
function [data, labels] = convertFeaturesToMatrix(features, chars)
    NUM_FEATURES = 19; % Update this if our features change
    data = zeros([length(features),NUM_FEATURES]);
    labels = zeros([1,length(features)]);
    for i = 1:length(features)
        obj = features{i};
        row = [obj.Area, obj.Centroid(1), obj.Centroid(2), obj.MajorAxisLength,...
            obj.MinorAxisLength, obj.Eccentricity, obj.Orientation, obj.ConvexArea,...
            obj.Circularity, obj.Solidity, obj.Perimeter, obj.HuMoments];
        class = strfind(chars,obj.Letter);
        data(i,:) = row;
        if ~isempty(class)
            labels(i) = class;
        end
    end
end
