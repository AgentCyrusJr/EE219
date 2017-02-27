% problem 4
%get indices of 10-fold CV
N =length(data);
kfold =10;
lambda = 0.01;
cvFolds = crossvalind('Kfold',N,kfold);
avg_err = zeros(kfold,1);
for i=1:kfold
    testIdx = (cvFolds == i);
    trainIdx = ~testIdx;
    train_data = data(trainIdx,:);
    test_data = data(testIdx,:);
    R_binary_train = zeros( max(data(:,1)), max(data(:,2)) );
    W_rating_train = zeros( max(data(:,1)), max(data(:,2)) ); 
    R_binary_test = zeros( max(data(:,1)), max(data(:,2)) );
    W_rating_test = zeros( max(data(:,1)), max(data(:,2)) ); 
    for j=1 :length(train_data)
        uid = train_data(j,1);
        mid = train_data(j,2);
        rating = train_data(j,3);
        R_binary_train(uid,mid) = 1;
        W_rating_train(uid,mid) = rating;
    end
    for n=1:length(k)
    [U_train,V_train,numIter,tElapsed,finalResidual]=wnmfrule_regularization(R_binary_train,W_rating_train,k(n),lambda);
    R_pred = U_train*V_train;
    %R_pred = min(R_pred,5);
    U4{n,i} = U_train;
    V4{n,i} = V_train;
    finalResidual4{n,i} = finalResidual;
    Prediction_err = zeros(length(test_data),1);
    for j=1:length(test_data)
        uid = test_data(j,1);
        mid = test_data(j,2);
        rating = test_data(j,3);
        Prediction_err(j) = abs(R_pred(uid,mid)-rating);
    end
    %avg_err(i) = mean(Prediction_err);
    avg_err2{n,i}= mean(Prediction_err);
    end
end
%%
% computer the precison and recall corresponding to the different threshold
threshold = 1:0.01:5;
prec = zeros(1,length(threshold));  
rec = zeros(1,length(threshold));   
it=1;
kfold=10;
lambda = 0.1;
for m=1:kfold
    testIdx = (cvFolds == m);
    trainIdx = ~testIdx;
    train_data = data(trainIdx,:);
    test_data = data(testIdx,:);
    R_binary_test = zeros( max(data(:,1)), max(data(:,2)) );
    for j=1 :length(test_data)
        uid = test_data(j,1);
        mid = test_data(j,2);
        R_binary_test(uid,mid) = 1;
    end
    U=U4{1,m}; %U=U4{1,m} corresponding to k=10;U=U4{2,m} corresponding to k=50;U=U4{3,m} corresponding to k=100
    V=V4{1,m}; %V=V4{1,m} corresponding to k=10;V=V4{2,m} corresponding to k=50;V=V4{3,m} corresponding to k=100
    R_pred =  W_rating_train.*(U*V);
    tmp = R_binary_test - U*V;
    tmp = max(tmp, eps);
    square_error(m) = sum(sum(W_rating_train.*(tmp.^2)));
    it=1;
for t = threshold
    tp = 0;     
    fp = 0;     
    fn = 0;
    
    for i=1:length(test_data)
        uid = test_data(i,1);
        mid = test_data(i,2);
        rating = test_data(i,3);
        if (R_pred(uid,mid) >= t)
            if (rating >= 4)
                tp = tp + 1;
            else
                fp = fp + 1;
            end
        elseif (rating >= 4)
            fn = fn + 1;
        end
    end
    prec(it) = tp/(tp+fp);
    rec(it) = tp/(tp+fn);
   it = it + 1;
end
precc{1,m}=prec;
recc{1,m}=rec;

plot(rec,prec)
hold on
end
title('Precision versus Recall(Regularization wnmf,lambda=0.01,k=10)')%k=50,k=100
xlabel('Recall')
ylabel('Precision')
square_err_mean = mean(square_error(1:10))

%%
for any fixed k value,averge the 10 curves in 10 folds to one curve

reccc=zeros(1,401);
for i=1:(kfold-1)
    reccc=reccc+recc{1,i};
end
reccc=reccc/(kfold-1);

preccc=zeros(1,401);
for j=1:(kfold-1)
    preccc=preccc+precc{1,j};
end
preccc=preccc/(kfold-1);

plot(reccc,preccc)
title('Precison versus Recall(Regularization wnmf,10-fold,lambda=0.01,k=10)')
xlabel('Recall')
ylabel('Precision')

[recccsort,index] = sort(reccc);
precccsort=preccc(index);
area=trapz(recccsort,precccsort)
a = num2str(area);
s = strcat('Area under curve:  ',a);
%legend('Area under curve: %f',area);
h=legend(s);
set(h,'Fontsize',20);
%prec10=preccc;rec10=reccc;

%% plot three curves with precision vs Recall in k=10,50,100 
plot(rec10,prec10)
hold on
plot(rec50,prec50)
hold on
plot(rec100,prec100)
title('Precison versus Recall (Regularized wnmf,10-fold,lambda=1)')
xlabel('Recall')
ylabel('precison')
legend('k=10','k=50','k=100')
%% plot three curves with Recall vs threshold in k=10,50,100 
plot(threshold,rec10)
hold on
plot(threshold,rec50)
hold on
plot(threshold,rec100)
title('Recall vs Thredhold (Regularized wnmf,10-fold,lambda=1)')
xlabel('threshold')
ylabel('Recall')
legend('k=10','k=50','k=100')
%% plot three curves with Precision vs threshold in k=10,50,100
plot(threshold,prec10)
hold on
plot(threshold,prec50)
hold on
plot(threshold,prec100)
title('Precison vs Thredhold (Regularized wnmf,10-fold,lambda=1)')
xlabel('threshold')
ylabel('Precison')
legend('k=10','k=50','k=100')
