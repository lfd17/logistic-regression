S = load('data_logistic.mat');
dataset = S.z;

dataSize = length(dataset);
testDataSize = round(length(dataset) * 0.4);
testData = zeros(testDataSize, 4);

rng(0,'twister');
for i = 1:testDataSize
    randomIndex = round((dataSize - 1) * rand());
    disp(randomIndex);
    testData(i,1:3) = dataset(randomIndex, :);
    dataset(randomIndex, :) = [];
    dataSize = dataSize - 1;
end

testData

learningRate = 0.3;
epochNumber = 1000;

coef = gradientDescent(dataset, learningRate, epochNumber);

for i = 1:length(testData)
    testData(i,4) = round(predict(testData(i,:), coef));
end

