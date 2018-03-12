%% Initialization
T = [0.7 0.3; 
     0.3 0.7];
Ot = [0.9 0;
      0 0.2];
Of = [0.1 0;
      0 0.8];
f = [0.5 0.5]'; %% Initial probabilities



%% Part B1
e = [1 1];
[j,length] = size(e);
for i = 1:length
    if e(i) == 1
        f = Ot*T'*f;
    else
        f = Of*T'*f; 
    end
    alpha = 1/sum(f);
    f = alpha.*f;
end     

%% Part B2
e = [1 1 0 1 1];
[j,length] = size(e);
fprintf("| Day | True  | False |\n");
for i = 1:length
    if e(i) == 1
        f = Ot*T'*f;
    else
        f = Of*T'*f; 
    end
    alpha = 1/sum(f);
    f = alpha.*f;
    fprintf("|  %d  | %0.3f | %0.3f |\n", i, f(1), f(2));
end     



