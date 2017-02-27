% problem 2
%get indices of 10-fold CV
fid = fopen('./dataset/ml-100k/u.data');
if fid == -1
   disp('Cannot open the file');
   return;
else
   inputText = textscan(fid,'%d%d%d%d');
   uid = inputText{1,1};
   mid = inputText{1,2};
   rating = inputText{1,3};
end   
    data = [uid mid rating];
    k = [10 50 100];
N =length(data);
kfold =10;
cvFolds = crossvalind('Kfold',N,kfold);
avg_err = zeros(kfold,1);
for i=1:kfold
    testIdx = (cvFolds == i);
    trainIdx = ~testIdx;
    train_data = data(trainIdx,:);
    test_data = data(testIdx,:);
    R_train = NaN( max(data(:,1)), max(data(:,2)) );
   % W_train = zeros( max(data(:,1)), max(data(:,2)) ); 
    for j=1 :length(train_data)
        uid = train_data(j,1);
        mid = train_data(j,2);
        rating = train_data(j,3);
        R_train(uid,mid) = rating;
   %    W_train(uid,mid) = 1;
    end
    %%%%%%%%%%%%%%%%%%%
    for n=1:length(k)
    [U_train,V_train,numIter,tElapsed,finalResidual]=wnmfrule(R_train,k(n));
    R_pred = U_train*V_train;
    %R_pred = min(R_pred,5);
    U2{n,i} = U_train;
    V2{n,i} = V_train;
    finalResidual2{n,i} = finalResidual;
    Prediction_err = zeros(length(test_data),1);
    for j=1:length(test_data)
        uid = test_data(j,1);
        mid = test_data(j,2);
        rating = test_data(j,3);
        Prediction_err(j) = abs(R_pred(uid,mid)-rating);
    end
    avg_err2{n,i}= mean(Prediction_err);
    end
end
