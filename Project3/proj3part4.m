addpath('./nmfv1_4/');
load('./ml-100k/u.data');
data = u(:,1:3);
user = data(:,1);
movie =data(:,2);
rating = data(:,3);
%k=[10,50,100];
option = struct('iter',200);

N =length(data);
k =10;
cvFolds = crossvalind('Kfold',N,k);
avg_err = zeros(k,1);

R = zeros( max(data(:,1)), max(data(:,2)) );
W = zeros( max(data(:,1)), max(data(:,2)) ); 
for j=1 :length(data)
        uid = data(j,1);
        mid = data(j,2);
        rat = data(j,3);
        R(uid,mid) = rat;
        W(uid,mid) = 1;
end
%% Recommend
R_binary = zeros( max(data(:,1)), max(data(:,2)) );
W_rating = zeros( max(data(:,1)), max(data(:,2)) );
prec=zeros(k,1);
hit_rate = zeros(k,1);
false_alarm_rate = zeros(k,1);
for i=1:k
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
        rat = train_data(j,3);
        R_binary_train(uid,mid) = 1;
        W_rating_train(uid,mid) = rat;
    end
    for j=1:length(test_data)
        uid = test_data(j,1);
        mid = test_data(j,2);
        rat = test_data(j,3);
        R_binary_test(uid,mid) = 1;
        W_rating_test(uid,mid) = rat;          
    end
    
    user_num = length(R(:,1));
    L=5;
    threshold=3;
    top_L_precision = zeros(user_num,1);
    [U_b,V_b,numIter,tElapsed,finalResidual]=wnmfrule_regularization(R_binary_train,W_rating_train,100,0.1,option);
    %[U_b,V_b,numIter,tElapsed,finalResidual]=mywnmfrule(R_binary_train,10,W_rating_train,option);
    R_recons_b = R.*R_binary_test.* (U_b*V_b);
    %R_recons_b = W_rating_test.* (U_b*V_b);
    %xx = (abs(U_b*V_b-ones(max(data(:,1)), max(data(:,2)))));
    %R_recons_b = R_binary_test.*xx;

    %%
    hit_all=0;
    count=0;
    like_all=0;
    miss_all=0;
    dislike_all=0;
    for j=1:user_num
%             this_row = R_recons_b(j,:);
%             index0=find(this_row>0);
%             this_row = this_row(index0);
%             [~,index] = sort(this_row);
%             %top_L_pred = index(1:L);
            [~,index]=sort(R_recons_b(j,:),'descend');
            ratings = W_rating_test(j,:); % W_rating_test may have less than L ratings. Ignore the entries that do not have actual rating.
            %ratings = R(j,:);
            [ww,~] = sort(R_binary_test(j,:),'descend');
            %[ww,~] = sort(R(j,:),'descend');
            ww=ww(1:L);
            l = length(find(ww>0));
            ww=ww(1:l);
            top_l_pred = index(1:l);
            if l>0
                ratings(top_l_pred);
                count=count+l;
                hit_all=hit_all+length(find(ratings(top_l_pred) >threshold));
                miss_all=miss_all+length(find(ratings(top_l_pred) <=threshold));
                like_all=like_all+length(find(ratings>threshold));
                rr=W_rating_test(j,:);
                have_ratings = find(rr>0);           
                rr=rr(have_ratings);
                dislike_all=dislike_all+length(find(rr<=threshold));
            end
            %top_L_precision(k) = length(find(ratings(top_L_pred) >threshold))/L;     
    end
    prec(i) = 1.0*hit_all/count;
    hit_rate(i)=1.0*hit_all/like_all;
    false_alarm_rate(i)=1.0*miss_all/dislike_all;
end