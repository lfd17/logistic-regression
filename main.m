% Logistic Regression Classifier
% BLG454E - HW2
%
% Fools on Parade
% Burak Sayýn - 150130120
% Burak Karakan - 150130114

% Get the dataset.
S = load('data_logistic.mat');
dataset = S.z;

% Test and training data sizes.
dataSize = length(dataset);
testDataSize = round(length(dataset) * 0.3);
testData = zeros(testDataSize, 4);

% Construct the test dataset.
rng(0,'twister');
for i = 1:testDataSize
    randomIndex = round((dataSize - 1) * rand());
    testData(i,1:3) = dataset(randomIndex, :);
    dataset(randomIndex, :) = [];
    dataSize = dataSize - 1;
end

% Plot a test classification result on a new figure.
epochNumber = 700;
learningRate = 0.3;
[error, testData] = calculateErrorRateWithClassification(dataset, testData, epochNumber, learningRate);
figure;
gscatter(testData(:,1), testData(:,2), testData(:,4), 'br','xo');
xlabel('First Feature');
ylabel('Second Feature');
title('Test data classification | Epoch: 700 - Learning Rate: 0.3'); 

% Construct the arrays for the plotting.
epochNumbers = 50:+50:1000;
learningRates = 0.025:+0.025:1.0;
errors = zeros(length(epochNumbers), length(learningRates));

epochIndex = 1;
for epochNumber = epochNumbers
    learningRateIndex = 1;
    
    for learningRate = learningRates        
        [errors(epochIndex,learningRateIndex), testData] = calculateErrorRateWithClassification(dataset, testData, epochNumber, learningRate);
        learningRateIndex = learningRateIndex + 1;
    end
    epochIndex = epochIndex + 1;
end

% Plot the results.
figure;
surf(learningRates, epochNumbers, errors);
title('Error rate of the Logistic Regression Classifier')
xlabel('Learning Rate')
ylabel('Epoch Numbers')
zlabel('Percentage Error Rate');
