% This is a placeholder for our classifier code

%this is a test

%This is another test -shad

%% Use this section for reading in data: 
close all;clear all;clc; 
% Read in data set 1: 
pnameExtractedData = 'C:\Users\Shad\Documents\School\EN.525.670.81 Machine Learning for Signal Processing\handwriting-classifier\Features\';

load([pnameExtractedData 'train_features.mat']);

DataSet1 = featuresData;
Data1.HuMoments = [] ;
Data1.Labels =[]; 

Num_Pics = length(DataSet1);

for i = 1:Num_Pics 
    
    Data1.Labels{i} = DataSet1{i}.Letter;

    Data1.HuMoments(i,:) = DataSet1{i}.HuMoments;

end
Data1.Labels = categorical(Data1.Labels); 

% Seperate Data as test set or verification set with dividerand?: 
TrainRatio = 0.7;
ValRatio = 0.15;
testRatio = 0.15;

%[TrainInd,ValInd,TestInd] = dividerand(Num_Pics,TrainRatio,ValRatio,testRatio);
%Use the first half of the data for training.

% Read in Data set 2: 

% Generic Input Data: 

% Possible classes: 
    alphabet = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', ...
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','o','s'};
    Classes = categorical(alphabet);
%% Shad Initial Swag

% Create Network Layers: 
 layers = [
    featureInputLayer(8)          % Input layer for feature data with 8 features
    fullyConnectedLayer(500)       % Fully connected layer with 10 neurons
    reluLayer()                   % ReLU activation layer
    fullyConnectedLayer(100) 
    reluLayer()  
    fullyConnectedLayer(28)        % Fully connected layer with 5 neurons
    softmaxLayer()                % Softmax layer for classification
    %classificationLayer()        % Classification output layer
];
 %The classificationlayer was not working, it did not want to be the output
 %layer? 
 %Maybe can add class weights based on letter likelyness? 
 
net = layerGraph(layers);
%plot(net)
%analyzeNetwork(net)


Options = trainingOptions('rmsprop',...
    'ExecutionEnvironment', 'gpu', ...  
    'MiniBatchSize', 64, ...
    'MaxEpochs', 10);

XTrain = Data1.HuMoments(1:Num_Pics/2,:);
YTrain = Data1.Labels(1:Num_Pics/2);
XTrainGPU = gpuArray(XTrain);
YTrainGPU = gpuArray(YTrain);


netTrained = trainnet(XTrainGPU, YTrainGPU, layers, 'crossentropy', Options);

analyzeNetwork(netTrained)


%% Initial Testing
Result = predict(netTrained,Data1.HuMoments((Num_Pics/2)+1:end,:));
%Result = classify(netTrained,Data1.HuMoments((Num_Pics/2)+1:end,:));

Correct = 0; 
letterNumber = zeros(1,length(alphabet)); 

for i = 1:length(Result) 
    Index = find(max(Result(i,:)) == Result(i,:));
    Result_Character(i) = alphabet(Index); 
end

Result_Character = categorical(Result_Character);

accuracy = sum(Result_Character == Data1.Labels((Num_Pics/2)+1:end)) / Num_Pics
        
confusionchart(Data1.Labels((Num_Pics/2)+1:end),Result_Character)













