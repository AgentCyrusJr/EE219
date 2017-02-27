% problem 3
threshold = 1:0.01:5;
prec = zeros(1,length(threshold));  
rec = zeros(1,length(threshold));   
it=1;
kfold=10;    
for m=1:kfold
    testIdx = (cvFolds == m);
    trainIdx = ~testIdx;
    train_data = data(trainIdx,:);
    test_data = data(testIdx,:);
    
    U=U2{1,m}; %U=U2{1,m} corresponding to k=10;U=U2{2,m} corresponding to k=50;U=U2{3,m} corresponding to k=100
    V=V2{1,m}; %V=V2{1,m} corresponding to k=10;V=V2{2,m} corresponding to k=50;V=V2{3,m} corresponding to k=100
    R_pred = U*V;
    it=1;
    tmp = R_binary_test - U*V;
    tmp = max(tmp, eps);
    square_error(m) = sum(sum(W_rating_train.*(tmp.^2)));
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
title('Precison versus Recall(Unregularization wnmf,10-fold,k=10)')
xlabel('Recall')
ylabel('Precision')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for any fixed k value,averge the 10 curves in 10 folds to one curve
reccc=zeros(1,401);
for i=1:kfold
    reccc=reccc+recc{1,i};
end
reccc=reccc/kfold;

preccc=zeros(1,401);
for j=1:kfold
    preccc=preccc+precc{1,j};
end
preccc=preccc/kfold;

plot(reccc,preccc)
title('Precison versus Recall(Regularization wnmf,10-fold,lambda=1,k=100)')
xlabel('Recall')
ylabel('Precision')

%computer the area under curve
[recccsort,index] = sort(reccc);
precccsort=preccc(index);
area=trapz(recccsort,precccsort)
a = num2str(area);
s = strcat('Area under curve:  ',a);
%legend('Area under curve: %f',area);
h=legend(s);
set(h,'Fontsize',20);


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
