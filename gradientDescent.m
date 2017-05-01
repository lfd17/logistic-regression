function coefficients = gradientDescent(data, learningRate, epochCount)  
    dataLength = length(data);
    coefficientLength = length(data(1,:));
    
    coefficients = zeros(1, coefficientLength);
    
    for epoch = 1:epochCount
        for i = 1:dataLength
            
            % Get the prediction.
            row = data(i,:);
            prediction = predict(row, coefficients);
            error = row(3) - prediction;
            
            % Calculate the first coefficient, which is not dependent on any feature.
            coefficients(1) = coefficients(1) + learningRate * error * prediction * (1 - prediction);
            
            % Calculate the coefficients.
            for j = 2:coefficientLength
                coefficients(j) = coefficients(j) + learningRate * error * prediction * (1 - prediction) * row(j - 1);
            end 
        end
    end
end