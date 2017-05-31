%
% 
%
%   alpha,beta = {100, 50, 20, 10, 5, 1}
%   lambda {1, 0.1, 0.01, .0.001}
%   error = {1, 0.5, 0.05, 0.01, 0.001}
%
function [Z, A, W, b] = plr(X, Q)
% solve:
% 	min||f(WX + B)||_* + lambda(||A||^2_F + ||W||^2_F + ||b||^2_2)
%	s.t. ||Q - A*f(WX + b)||^2_F < tol
%     
%   f:  sigmoid function
%   W: projection matrix to be learned
%   b: bias
%	B: each colum is b
% 	A: the overcompleted label matrix
%	Z: Z = f(WX + B)
%
%	X: input samples, d x n, d is dimension
%	Q: label matix of input samples 
% 

alpha = 5;
beta = 10;
lambda = 0.1;
epsilon = 0.01;

tol = 0.01;
dictSize= 500; % label dictionary atom number

r = dictSize;

iter = 1;
MaxIter = 50;

[d, n] = size(X);
%初始化偏差和投影矩阵
b = zeros(r, 1);
W = normrnd(0, 0.01, [r, d]);
sv =10;
I = eye(r, r);

opt.tol = tol;%precision for computing the partial SVD
opt.p0 = ones(n,1);


while iter < MaxIter
    %step1, 构建偏差矩阵
    B = repmat(b, 1, n);
    
    % 计算F， F = f(WX+B)
    Temp = W * X + B;
    F = sigmoid(Temp);
    
    %计算Z , by SVT
    [Z, sv] = svt(F, n, alpha, sv, opt);

    
    %计算 A, A = QF'(FF' + lambda\beta I)^-1
    temp = F*F' + lambda/ beta * I;
    A = Q * F' * inv(temp);
    
    %计算W 和 b
    % sigmoid: f(a) x(1 - f(a))
    dF = F .*(ones(size(F)) - F);
    
    Temp1 = dF.*(A' * A * F - A' * Q);
    Temp2 = alpha / beta *(dF.*(Z- F));
    
    dW = 2*(Temp1* X') + 2 *(Temp2*X') + 2*lambda / beta * W;
    W = W - epsilon * dW;
    
    Temp3 = Temp1 + Temp2;
    db = 2* sum(Temp3, 2) + 2*lambda / beta *b;
    b = b - epsilon * db;
    
    if norm(Q - A* sigmoid(W * X + B), inf) < tol
        break;
    end

    clear B;
    iter = iter + 1;
end


end
