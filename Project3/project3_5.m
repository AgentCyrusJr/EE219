% load the data from ratings.csv
data = csvread('C:\Users\A.Cleverley\Desktop\LA\course\EL ENGR 219\EE219\Project3\ml-latest-small\ml-latest-small\ratings.csv',1);
user_id = data(:, 1);
movie_id = data(:, 2);
rating = data(:, 3);

max_user_id = max(user_id);
max_movie_id = max(movie_id);
% construct R, W
R = zeros(max_user_id, max_movie_id);
W = zeros(max_user_id, max_movie_id);

shape = size(data);
for i = 1:shape(1)
    R(user_id(i), movie_id(i)) = rating(i);
    W(user_id(i), movie_id(i)) = 1;
end

n_cv = 10; 
precision = zeros(n_cv);
L = 1;
N = size(movie_id);
N = N(1);

index = randperm(N)';
for i = 0 : 9
    index_i = index(i*(N/10)+1:  (i+1)*(N/10));
    R_i= R;
    W_i = W;
    for j = 1:size(index_i)
        R_i(user_id(index_i(j)), movie_id(index_i(j))) = 0;
        W_i(user_id(index_i(j)), movie_id(index_i(j))) = 1;
    end
    [U_i, V_i] = wnmfrule(R_i, 10, 0.01);
    R_i_predict = U_i * V_i;
    R_i_predict = R_i_predict.*R.*W_i;
    [B, I] = sort(R_i_predict, 2, 'descend');
    
    top_L_recommend = zeros(max_user_id, max_movie_id);
    for j = 1:max_user_id
        for k = 1:L
            top_L_recommend(j, I(j, k)) = 1;
        end
    end
    
    prediction  = top_L_recommend .* R_i_predict;
    total = 0; tp =0;
    for j = 1:max_user_id
        for k = 1:max_movie_id
           if prediction(j, k) > 0
               total = total + 1;
               if R(j, k) >= 4
                   tp = tp + 1;
               end
           end
        end
    end
    precision(i+1) = tp/total;
end
fprintf('The average precision is %f', sum(precision)/10);








