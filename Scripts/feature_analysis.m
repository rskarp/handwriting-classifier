% Feature Analysis
% Generate histograms showing distribution of letters
% Feature Ranking

%% Load Feature Data
close all; clear;
trainFeatures = load_features("train");
validationFeatures = load_features("validation");
testFeatures = load_features("test");

%% Generate Histograms
% Generate histograms showing distribution of letters in train, validation,
% and test datasets.

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

%% Feature Ranking

% Convert features to matrix
chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
normalize = 1;
[trainData, trainLabels, trainLetters, trainLabelMatrix] = ...
    convertFeaturesToMatrix(trainFeatures, chars, normalize);

warning('off','MATLAB:singularMatrix')
warning('off','MATLAB:nearlySingularMatrix')

bestFeaturesMatrix = zeros([26 26 5]);
bestFeaturesCounts = zeros([1 size(trainData,2)]);
for c1=1:length(chars)
    for c2=c1+1:length(chars)
        % Use proper class data
        class1 = trainData(trainLabels==c1,:);
        class2 = trainData(trainLabels==c2,:);
        nPts = min(size(class1,1),size(class2,1));
        class1 = class1(1:nPts,:)';
        class2 = class2(1:nPts,:)';
        % Salar feature ranking
        [T]=ScalarFeatureSelectionRanking(class1,class2,'divergence');
        % Rank the features using the features using cross-correlation
        a1=0.2;a2=0.8;
        [p]= compositeFeaturesRanking(class1,class2,a1,a2,T);
        % Print out the Composite Scalar Feature Selection results 
        fprintf('\n Class %s-%s \n',chars(c1),chars(c2));
        fprintf('  Composite Scalar Feature Ranking:');
        for i=1:size(p,1)
            fprintf(' (%d)',p(i));
        end
        fprintf('\n');

        % Vector Feature Evaluation using Exhaustive Search with ScatterMatrices
        inds=sort(p,'ascend');
        [cLbest,Jmax]=exhaustiveSearch(class1(inds,:),class2(inds,:),'ScatterMatrices',5);

        % Print the Exhaustive Search results
        ids = inds(cLbest);
        fprintf('  Exhaustive Search -> Best of four: ');
        fprintf(' (%d)',ids);
        fprintf('\n');

        bestFeaturesCounts(ids) = bestFeaturesCounts(ids)+1;

        bestFeaturesMatrix(c1,c2,:) = ids;
    end
end

features = {'Area', 'Centroid(1)', 'Centroid(2)', 'MajorAxisLength',...
            'MinorAxisLength', 'Eccentricity', 'Orientation', 'ConvexArea',...
            'Area/ConvexArea','MajorAxisLength/MinorAxisLength',...
            'Circularity', 'Solidity', 'Perimeter', 'Hu1', 'Hu2', 'Hu3',...
            'Hu4', 'Hu5', 'Hu6', 'Hu7', 'Hu8'};
[~,I] = maxk(bestFeaturesCounts,10);

top10Feats = features(I)'
