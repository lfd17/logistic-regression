% Logistic Regression Classifier
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

epochNumbers = 50:+50:1000;
learningRates = 0.0:+0.05:1.0;
errors = zeros(length(epochNumbers), length(learningRates));
length(epochNumbers)
length(learningRates)
size(errors)


epochIndex = 1;
for epochNumber = epochNumbers
    learningRateIndex = 1;
    for learningRate = learningRates
        
        % Calculate the coefficients using stochastic gradient descent.
        coef = gradientDescent(dataset, learningRate, epochNumber);

        % Construct test predictions.
        errorRate = 0;
        for i = 1:length(testData)
            testData(i,4) = round(predict(testData(i,:), coef));
            if testData(i,3) ~= testData(i,4)
                errorRate = errorRate + 1;
            end
        end

        errorRate = (errorRate / testDataSize) * 100;
        errors(epochIndex,learningRateIndex) = errorRate;
        fprintf('Epoch Count: %d - Learning Rate: %.2f - Error: %f\n', epochNumber, learningRate, errorRate);
        learningRateIndex = learningRateIndex + 1;
    end
    epochIndex = epochIndex + 1;
end

surf(learningRates, epochNumbers, errors);
title('Error rate of the Logistic Regression Classifier')
xlabel('Learning Rate')
ylabel('Epoch Numbers')
zlabel('Percentage Error Rate');
