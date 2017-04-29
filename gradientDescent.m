function coefficients = gradientDescent(data, learningRate, epochCount)  
    dataLength = length(data);
    coefficientLength = length(data(1,:));
    
    coefficients = zeros(1, coefficientLength);
    
    for epoch = 1:epochCount
        squaredError = 0;
        for i = 1:length(data)
            row = data(i,:);
            prediction = predict(row, coefficients);
            error = row(3) - prediction;
            squaredError = squaredError + error^2;
            coefficients(1) = coefficients(1) + learningRate * error * prediction * (1 - prediction);
            
            for j = 2:coefficientLength
                coefficients(j) = coefficients(j) + learningRate * error * prediction * (1 - prediction) * row(j - 1);
            end 
        end
    end
end