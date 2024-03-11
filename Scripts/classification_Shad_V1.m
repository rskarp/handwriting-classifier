% This is a placeholder for our classifier code

%this is a test

%This is another test -shad

%% Use this section for reading in data: 
close all;clear all;clc; 
% Read in data set 1: 
pnameExtractedData = 'C:\Users\Shad\Documents\School\EN.525.670.81 Machine Learning for Signal Processing\handwriting-classifier\Features\';

load([pnameExtractedData 'train_features.mat']);
Test.Data = featuresData;

load([pnameExtractedData 'test_features.mat']);
Val.Data = featuresData;

load([pnameExtractedData 'validation_features.mat']);
Train.Data = featuresData;

Test.Num_Pics = length(Test.Data);
Train.Num_Pics = length(Train.Data);
Val.Num_Pics = length(Val.Data);

%%
ReduceIndex = 0;
for i = 1:Test.Num_Pics
    if Test.Data{i}.Letter == 'o' || Test.Data{i}.Letter == 's' 
        %Test.Num_Pics = Test.Num_Pics -1; 
        fprintf('----Notify----\n\n')
        ReduceIndex = ReduceIndex+1; 
    else
        Test.Labels{i-ReduceIndex} = Test.Data{i}.Letter;
        Test.HuMoments(i-ReduceIndex,:) = Test.Data{i}.HuMoments;
    end
end

for i = 1:Train.Num_Pics
    Train.Labels{i} = Train.Data{i}.Letter;
    Train.HuMoments(i,:) = Train.Data{i}.HuMoments;
end

ReduceIndex = 0;
for i = 1:Val.Num_Pics
    if Val.Data{i}.Letter == 'a' || Val.Data{i}.Letter == 'd' || Val.Data{i}.Letter == 'e' || Val.Data{i}.Letter == 'i' || Val.Data{i}.Letter == 'o' || Val.Data{i}.Letter == 't' 
        %Val.Num_Pics = Val.Num_Pics -1; 
        fprintf('----Notify----\n\n')
        ReduceIndex = ReduceIndex+1; 
    else
        Val.Labels{i-ReduceIndex} = Val.Data{i}.Letter;
        Val.HuMoments(i-ReduceIndex,:) = Val.Data{i}.HuMoments;
    end
end

%% Convert data into catagories: 
Train.Labels = categorical(Train.Labels); 
Test.Labels = categorical(Test.Labels); 
Val.Labels = categorical(Val.Labels); 

% Generic Input Data: 

% Possible classes: 
    alphabet = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', ...
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
    Classes = categorical(alphabet);
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
    'MaxEpochs', 15,...
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
        
confusionchart(Test.Labels,Result_Character)












