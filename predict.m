function prediction = predict(row, coefficients)
    prediction = coefficients(1);
    for i = 2:3
        prediction = prediction + coefficients(i) * row(i - 1);
    end
    
    % Calculate the prediction result.
    prediction = round(1.0 / (1.0 + exp(-prediction)), 3);
end