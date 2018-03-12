%% Initialization
T = [0.7 0.3; 
     0.3 0.7];
Ot = [0.9 0;
      0 0.2];
Of = [0.1 0;
      0 0.8];
f0 = [0.5;0.5]; %% Initial probabilities

%% Part C 1
%Initialize the variables needed for this problem
e = [1 1];
[j,length] = size(e);
fv = zeros(2, length+1); 
b = [1;1];
sv = zeros(2, length+1);

fv(:,1) = f0;

%Do the forwards filtering
for i = 1:length
    if e(i) == 1
        fv(:,i+1) = Ot*T'*fv(:,i);
    else
        fv(:,i+1) = Of*T'*fv(:,i); 
    end
    fv(:,i+1) = normalize(fv(:,i+1));
end

%Do the backwards smoothing
for i = length:-1:1
    sv(:,i+1) = normalize((fv(:,i+1).*b));
    if e(i) == 1
        b = T*Ot*b;
    else
        b = T*Of*b; 
    end
end
%Print the result
fprintf("Probabilities for rain at day 1: \n")
fprintf("True: %.3f\n", sv(1,2));
fprintf("False: %.3f\n", sv(2,2));


%% Part C 2
%initialize needed values for this problem
e = [1 1 0 1 1];
[j,length] = size(e);
fv = zeros(2, length+1); 
b = [1;1];
sv = zeros(2, length+1);

fv(:,1) = f0;
%Do the forwards filtering
for i = 1:length
    if e(i) == 1
        fv(:,i+1) = Ot*T'*fv(:,i);
    else
        fv(:,i+1) = Of*T'*fv(:,i); 
    end
    fv(:,i+1) = normalize(fv(:,i+1));
end

%Do the backwards smoothing
for i = length:-1:1
    sv(:,i+1) = normalize((fv(:,i+1).*b));
    if e(i) == 1
        b = T*Ot*b;
    else
        b = T*Of*b; 
    end
end

%Print the results
fprintf("| Day | True  | False |\n");
for i = 2:length+1
    fprintf("|  %d  | %0.3f | %0.3f |\n", i-1, sv(1,i), sv(2,i));
end



