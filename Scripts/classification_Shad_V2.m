% V2 - 3/18/24 
%   Read in the rest of the training/ test/ and verification
%   data. Continued to refine network structure.

%% Use this section for reading in data: 
close all;clear all;clc; 
% Read in data set 1: 
pnameExtractedData.test = 'C:\Users\Shad\Documents\School\EN.525.670.81 Machine Learning for Signal Processing\handwriting-classifier\Features\test\';
pnameExtractedData.train = 'C:\Users\Shad\Documents\School\EN.525.670.81 Machine Learning for Signal Processing\handwriting-classifier\Features\train\';
pnameExtractedData.validate = 'C:\Users\Shad\Documents\School\EN.525.670.81 Machine Learning for Signal Processing\handwriting-classifier\Features\validation\';

Test.Data = []; 
Train.Data = []; 
Val.Data = []; 

for i = 1:3
    load([pnameExtractedData.test 'test_features_' num2str(i) '.mat']);
    Test.Data = [Test.Data data];
end

for i = 1:3
    load([pnameExtractedData.validate 'validation_features_' num2str(i) '.mat']);
    Val.Data = [Val.Data data];
end

for i = 1:5
    load([pnameExtractedData.train 'train_features_' num2str(i) '.mat']);
    Train.Data = [Train.Data data];
end

Test.Num_Pics = length(Test.Data);
Train.Num_Pics = length(Train.Data);
Val.Num_Pics = length(Val.Data);

%% Reorganize the data: 

for i = 1:Test.Num_Pics
    Test.Labels{i} = Test.Data{i}.Letter;
    Test.HuMoments(i,:) = Test.Data{i}.HuMoments;
end

for i = 1:Train.Num_Pics
    Train.Labels{i} = Train.Data{i}.Letter;
    Train.HuMoments(i,:) = Train.Data{i}.HuMoments;
end

for i = 1:Val.Num_Pics
    Val.Labels{i} = Val.Data{i}.Letter;
    Val.HuMoments(i,:) = Val.Data{i}.HuMoments;
end

%% Convert data into catagories: 
Train.Labels = categorical(Train.Labels); 
Test.Labels = categorical(Test.Labels); 
Val.Labels = categorical(Val.Labels); 


% Possible classes: 
    alphabet = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', ...
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
    Classes = categorical(alphabet);

    if length(Classes) ~= length(unique(Train.Labels)) || length(Classes) ~= length(unique(Test.Labels)) || length(Classes) ~= length(unique(Val.Labels))
        error('ERROR: The data has more labels than the labels.\n\n')
    end
%% Create Nueral Network 

% Create Network Layers: 
 layers = [
    featureInputLayer(8)          % Input layer for feature data with 8 features
    fullyConnectedLayer(500)          % ReLU activation layer
    reluLayer()  
    fullyConnectedLayer(100) 
    reluLayer()  
    fullyConnectedLayer(26)        % Fully connected layer with 5 neurons
    softmaxLayer()                % Softmax layer for classification
    %classificationLayer()        % Classification output layer
];
 %The classificationlayer was not working, it did not want to be the output
 %layer? 
 %Maybe can add class weights based on letter likelyness? 
 
net = layerGraph(layers);
%plot(net)
%analyzeNetwork(net)

XTrain = Train.HuMoments;
YTrain = Train.Labels;

XValidation = Val.HuMoments;
YValidation = Val.Labels;


% If you do not have GPU, comment this out: 
    XTrain = gpuArray(XTrain);
    Options = trainingOptions('rmsprop',...
    'ExecutionEnvironment', 'gpu', ...  
    'MiniBatchSize', 64, ...
    'MaxEpochs', 7,...
    'ValidationData', {XValidation, YValidation'}, ...
    'ValidationFrequency', 500, ...  % Validate every 5 epochs
    'Verbose', true,...
    Plots="training-progress",...
    OutputNetwork = "best-validation-loss",...
    InitialLearnRate = 0.01...
    );  % Display training progress
     
%{
   Else 
    Options = trainingOptions('rmsprop',...
    'MaxEpochs', 30,...
    'ValidationData', {XValidation, YValidation}, ...
    'ValidationFrequency', 5, ...  % Validate every 5 epochs
    'Verbose', true,...  % Display training progress
    );
%}
   
netTrained = trainnet(XTrain, YTrain', layers, 'crossentropy', Options);

analyzeNetwork(netTrained)

%% Initial Testing
Result = predict(netTrained,Test.HuMoments);
%Result = classify(netTrained,Data1.HuMoments((Num_Pics/2)+1:end,:));

Correct = 0; 
letterNumber = zeros(1,length(alphabet)); 

for i = 1:length(Result) 
    Index = find(max(Result(i,:)) == Result(i,:));
    Result_Character(i) = alphabet(Index); 
end

Result_Character = categorical(Result_Character);

accuracy = sum(Result_Character == Test.Labels) / Test.Num_Pics

figure()
confusionchart(Test.Labels,Result_Character)
title('Confusion Chart -  Hu Moments Only')












