% problem 1 

% create matrices R and W
    fid = fopen('./ml-100k/u.data');
    if fid == -1
       disp('Cannot open the file');
       return;
    else
      inputText = textscan(fid,'%d%d%d%d');
      uid = inputText{1,1};
      mid = inputText{1,2};
      rating = inputText{1,3};
    
    data = [uid mid rating];
    %W = zeros(max(data(:,1)), max(data(:,2)));
    R = NaN(max(data(:,1)), max(data(:,2)));
        
    for i=1:length(data)
        R(uid(i),mid(i)) = rating(i);
        W(uid(i),mid(i)) = 1;
    end    
    end
% Weighted Factorization
k = [10 50 100];
for i=1:length(k)
[U,V,numIter,tElapsed,finalResidual]=wnmfrule(R,k(i));
U1{1,i} = U;
V1{1,i} = V;
finalResidual1{1,i} = finalResidual;
%compute square error
tmp = R - U*V;
tmp = max(tmp, eps);
square_error = sum(sum(W.*(tmp.^2)));
square_error1{1,i} = square_error;
clear tmp;
end
