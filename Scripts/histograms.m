% Generate histograms showing distribution of letters in train, validation,
% and test datasets.

%% Load Feature Data
close all; clear;
trainFeatures = load_features("train");
validationFeatures = load_features("validation");
testFeatures = load_features("test");

%% Generate Histograms
% Get letter labels
trainLetters = cellfun(@(x) x.Letter,trainFeatures);
validationLetters = cellfun(@(x) x.Letter,validationFeatures);
testLetters = cellfun(@(x) x.Letter,testFeatures);
allLetters = strcat(trainLetters,validationLetters,testLetters);

% Categorize letter labels
C_train = categorical(num2cell(trainLetters));
C_validation = categorical(num2cell(validationLetters));
C_test = categorical(num2cell(testLetters));
C_all = categorical(num2cell(allLetters));

% Plot Histograms - Percents
figure; sgtitle('Letter Distributions (Percent)');
subplot(2,2,1); histogram(C_train,'Normalization','probability');
title('Train Data'); ylabel('Percent'); yticklabels(yticks*100);

subplot(2,2,2); histogram(C_validation,'Normalization','probability');
title('Validation Data'); ylabel('Percent'); yticklabels(yticks*100);

subplot(2,2,3); histogram(C_test,'Normalization','probability');
title('Test Data'); ylabel('Percent'); yticklabels(yticks*100);

subplot(2,2,4); histogram(C_all,'Normalization','probability');
title('Combined Data'); ylabel('Percent'); yticklabels(yticks*100);

% Plot Histograms - Counts
figure; sgtitle('Letter Distributions (Counts)');
subplot(2,2,1); histogram(C_train);
title('Train Data'); ylabel('Count');

subplot(2,2,2); histogram(C_validation);
title('Validation Data'); ylabel('Count');

subplot(2,2,3); histogram(C_test);
title('Test Data'); ylabel('Count');

subplot(2,2,4); histogram(C_all);
title('Combined Data'); ylabel('Count');
