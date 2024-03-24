% V2 - 3/18/24 
%   Read in the rest of the training/ test/ and verification
%   data. Continued to refine network structure.

% V3 - 3/18/24
%   Added more data in parallel with Hu moments

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

for i = 1:17 % Make this number less if less data
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
    Test.Ellispe(i) = Test.Data{i}.MajorAxisLength / Test.Data{i}.MinorAxisLength;
    Test.Eccentricity(i) = Test.Data{i}.Eccentricity;
    Test.Orientation(i) = Test.Data{i}.Orientation;
    Test.ConvexArea(i) = Test.Data{i}.ConvexArea;
    Test.Circularity(i) = Test.Data{i}.Circularity;
    Test.Solidity(i) = Test.Data{i}.Solidity;
    Test.Perimeter(i) = Test.Data{i}.Perimeter;
end

for i = 1244535:Train.Num_Pics
    Train.Labels{i} = Train.Data{i}.Letter;
    Train.HuMoments(i,:) = Train.Data{i}.HuMoments;
    Train.Ellispe(i) = Train.Data{i}.MajorAxisLength / Train.Data{i}.MinorAxisLength;
    Train.Eccentricity(i) = Train.Data{i}.Eccentricity;
    Train.Orientation(i) = Train.Data{i}.Orientation;
    Train.ConvexArea(i) = Train.Data{i}.ConvexArea;
    Train.Circularity(i) = Train.Data{i}.Circularity;
    Train.Solidity(i) = Train.Data{i}.Solidity;
    Train.Perimeter(i) = Train.Data{i}.Perimeter;
end

for i = 1:Val.Num_Pics
    Val.Labels{i} = Val.Data{i}.Letter;
    Val.HuMoments(i,:) = Val.Data{i}.HuMoments;
    Val.Ellispe(i) = Val.Data{i}.MajorAxisLength / Val.Data{i}.MinorAxisLength;
    Val.Eccentricity(i) = Val.Data{i}.Eccentricity;
    Val.Orientation(i) = Val.Data{i}.Orientation;
    Val.ConvexArea(i) = Val.Data{i}.ConvexArea;
    Val.Circularity(i) = Val.Data{i}.Circularity;
    Val.Solidity(i) = Val.Data{i}.Solidity;
    Val.Perimeter(i) = Val.Data{i}.Perimeter;
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
    featureInputLayer(15)          % Input layer for feature data with 8 features
    fullyConnectedLayer(100)          % ReLU activation layer
    reluLayer()  
    fullyConnectedLayer(50) 
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

XTrain = horzcat(Train.HuMoments,Train.Ellispe', Train.Eccentricity' ,Train.Orientation',Train.ConvexArea',...
    Train.Circularity',Train.Solidity',Train.Perimeter');

YTrain = Train.Labels;

XValidation = horzcat(Val.HuMoments,Val.Ellispe', Val.Eccentricity' ,Val.Orientation',Val.ConvexArea',...
    Val.Circularity',Val.Solidity',Val.Perimeter');
YValidation = Val.Labels;


% If you do not have GPU, comment this out: 
    XTrain = gpuArray(XTrain);
    Options = trainingOptions('rmsprop',...
    'ExecutionEnvironment', 'gpu', ...  
    'MiniBatchSize', 128, ...
    'MaxEpochs', 10,...
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
Result = predict(netTrained,horzcat(Test.HuMoments,Test.Ellispe', Test.Eccentricity' ,Test.Orientation',Test.ConvexArea',...
    Test.Circularity',Test.Solidity',Test.Perimeter'));
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
title('Confusion Chart -  All Image Metrics')

save('Net_V3p1.mat','netTrained')










