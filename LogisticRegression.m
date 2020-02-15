data = csvread('diabetes2.csv');
X = data(2:700, 2:3); y = data(2:700, 9);

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Glucose')
ylabel('Blood pressure')

% Specified in plot order
legend('Diabetes', 'Not diabetes')
hold off;
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n+1 , 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

options = optimset('GradObj', 'on', 'MaxIter', 500);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
  printf("%f",theta);
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Glucose')
ylabel('Blood pressure')
% Specified in plot order
legend('Diabetes', 'Not Diabetes')
hold off;
  p = predict(theta, X);
prob = sigmoid([1 88 60] * theta);
fprintf([' %f\n'], prob);
prob1 = sigmoid([1 115 69.1] * theta);
fprintf([' %f\n'], prob1);         
prob2= sigmoid([1 125 70] * theta);
fprintf([' %f\n'], prob2);
prob3 = sigmoid([1 95 70] * theta);
fprintf([' %f\n'], prob3);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (approx): 89.0\n');
fprintf('\n');
