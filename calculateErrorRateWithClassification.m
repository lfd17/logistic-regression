function [errorRate, testData] = calculateErrorRateWithClassification(dataset, testData, epochNumber, learningRate)

    testDataSize = length(testData);
    % Calculate the coefficients using stochastic gradient descent.
    coef = gradientDescent(dataset, learningRate, epochNumber);

    % Construct test predictions.
    errorRate = 0;
    for i = 1:testDataSize
        testData(i,4) = round(predict(testData(i,:), coef));
        if testData(i,3) ~= testData(i,4)
            errorRate = errorRate + 1;
        end
    end

    errorRate = (errorRate / testDataSize) * 100;
    fprintf('Epoch Count: %d - Learning Rate: %.3f - Error: %f\n', epochNumber, learningRate, errorRate);
end

