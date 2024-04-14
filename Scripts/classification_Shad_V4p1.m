% V2 - 3/18/24 
%   Read in the rest of the training/ test/ and verification
%   data. Continued to refine network structure.

% V3 - 3/18/24
%   Added more data in parallel with Hu moments

%% Use this section for reading in data: 
close all;clear all;clc; 
% Read in data set 1: 

addpath('C:\Users\Shad\Documents\School\EN.525.670.81 Machine Learning for Signal Processing\handwriting-classifier\Functions')

pnameExtractedData.test = 'C:\Users\Shad\Documents\School\EN.525.670.81 Machine Learning for Signal Processing\handwriting-classifier\Features\test\';
pnameExtractedData.train = 'C:\Users\Shad\Documents\School\EN.525.670.81 Machine Learning for Signal Processing\handwriting-classifier\Features\train\';
pnameExtractedData.validate = 'C:\Users\Shad\Documents\School\EN.525.670.81 Machine Learning for Signal Processing\handwriting-classifier\Features\validation\';

Test.Data = []; 
Train.Data = []; 
Val.Data = []; 

Test.Load = load_features("test");
Train.Load = load_features("train");
Val.Load = load_features("validation");

Test.Num_Pics = length(Test.Data);
Train.Num_Pics = length(Train.Data);
Val.Num_Pics = length(Val.Data);

%% Reorganize the data: 
chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

[Train.data, Train.labels, Train.letters, Train.labelMatrix] = convertFeaturesToMatrix(Train.Load, chars, 0);
[Test.data, Test.labels, Test.letters, Test.labelMatrix] = convertFeaturesToMatrix(Test.Load, chars, 0);
[Val.data, Val.labels, Val.letters, Val.labelMatrix] = convertFeaturesToMatrix(Val.Load, chars, 0);


% Change a hyper parameter: 
    Parameter.Name = 'MaxEpochs';
    Parameter.Value = [50];

%% Create Nueral Network 
[~,numFeatures] = size(Train.data);
numClasses = length(unique(Train.labels));

for ChangeParameter = 1:length(Parameter.Value)  
    % Create Network Layers: 
    layers = [
        featureInputLayer(numFeatures,Normalization="zscore")
        fullyConnectedLayer(200)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(100)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(50)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer];
    
     
    net = layerGraph(layers);
    %plot(net)
    %analyzeNetwork(net)
   
    
    
    % If you do not have GPU, comment this out: 
        XTrain = gpuArray(Train.data);
        
        Options = trainingOptions('adam',...
            MaxEpochs=50,...
            MiniBatchSize=4096, ...
            LearnRateSchedule='piecewise',...
            Metrics='accuracy',...
            ValidationPatience=5,...
            Shuffle='every-epoch',...
            ValidationData={Val.data, Val.labels'}, ...
            ValidationFrequency=500, ...  % Validate every 500 iterations
            Plots="training-progress",...
            OutputNetwork = "best-validation-loss",...
            InitialLearnRate = 0.01,...
            ExecutionEnvironment = 'gpu');  

    %{
       Options = trainingOptions('adam',...
            MaxEpochs=50,...
            MiniBatchSize=4096, ...
            LearnRateSchedule='piecewise',...
            Metrics='accuracy',...
            ValidationPatience=5,...
            Shuffle='every-epoch',...
            ValidationData={validationData, categorical(validationLetters)'}, ...
            ValidationFrequency=500, ...  % Validate every 500 iterations
            Plots="training-progress",...
            OutputNetwork = "best-validation-loss",...
            InitialLearnRate = 0.01...
            'ExecutionEnvironment', 'gpu');  
    %}
       
    netTrained{ChangeParameter} = trainnet(Train.data, Train.labels, layers, 'crossentropy', Options);
    
    analyzeNetwork(netTrained{ChangeParameter})
end

%% Choose one (highlight and press f9): 

    % save('Net_V4p0.mat','netTrained');
    % load('Net_V4p0.mat');
    
%% Initial Testing
close all
for ChangeParameter = 1:length(Parameter.Value)
    
    %Result = predict(netTrained{ChangeParameter},horzcat(Test.HuMoments,Test.Ellispe', Test.Eccentricity' ,Test.Orientation',Test.ConvexArea',...
    %    Test.Circularity',Test.Solidity',Test.Perimeter'));
    %Result = classify(netTrained,Data1.HuMoments((Num_Pics/2)+1:end,:));

    Result = predict(netTrained{ChangeParameter},Test.HuMoments);
    
    for i = 1:length(Result) 
        Index = find(max(Result(i,:)) == Result(i,:),1);
        Result_Character(i) = alphabet(Index); 
    end 
    
    Result_Character = categorical(Result_Character);
    
    accuracy = sum(Result_Character == Test.Labels) / Test.Num_Pics;
    
    figure()
    confusionchart(Test.Labels,Result_Character)
    title(['Confusion Chart: ' Parameter.Name ' = ' num2str(Parameter.Value(ChangeParameter)) ' , Accuracy = ' num2str(accuracy)] )
    %set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0., 1, 0.96]);
end










