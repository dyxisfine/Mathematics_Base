close all;clear;clc;
% load('ticai.mat')
% load('ticai01.mat')
load('ticai20004.mat') 
% 
% nnn = [nnn;17,20,21,29,30,5,9];
% return
CountFlag = 1;
nnn_count = 2;
nnn = nnn(1:end,:);

% AimY = nnn(:,3);
% figure
% autocorr(AimY)
% figure
% parcorr(AimY)
% return
[Summary, ~] = size(nnn);
        FowwardRW = 1./(1+exp(-ones(1,35)*(1/35)));
        BackwardRW = 1./(1+exp(-ones(1,12)*(1/12)));
while CountFlag>0
         BiasF =  nnn(nnn_count,1:5);
         BiasB =  nnn(nnn_count,6:7);
         [ Bias_usedF ] = SampleExpanding( BiasF, 35);
         [ Bias_usedB ] = SampleExpanding( BiasB, 12);
             Current_Bias = [1./(1+exp(-Bias_usedF)), 1./(1+exp(-Bias_usedB))];
             output_layer = ([FowwardRW,BackwardRW]) + 1.8 * Current_Bias;
        ForwardRW =output_layer(1,1:35);
        BackwardRW = output_layer(1,36:47);
         
mid_nnn = nnn(1:nnn_count,:);    
PeriodNum =2;
[PeriodSum, ~] = size(mid_nnn );

Forward = mid_nnn (:,1:5);
Backward = mid_nnn  (:,6:7);
 
count_num = 2;
flag=1;
Record = zeros(PeriodSum,1);
while flag>0
    count =0;
for i=1:5
    for j=1:5
       if  Forward(count_num,i)== Forward(count_num-1,j)
           count = count +1;
       end
    end
end
Record(count_num) = count;
count_num = count_num +1;
    if count_num > PeriodSum-1
        flag =0;
    end
end

P_1_1 = length(find (Record ==1))/(PeriodSum-1); 
P_2_1 = length(find (Record ==2))/(PeriodSum-1); 
P_3_1 = length(find (Record ==3))/(PeriodSum-1); % nchoosek(n,m)
P_1Hat =( nchoosek(35,4)* nchoosek(5,1) * nchoosek(34,4) )/( nchoosek(35,5)^(2) );
P_2Hat = ( nchoosek(35,3)* nchoosek(5,2) * nchoosek(33,3) )/( nchoosek(35,5)^(2) );
P_3Hat = ( nchoosek(35,2)* nchoosek(5,3) * nchoosek(32,2) )/( nchoosek(35,5)^(2) );
mid_used = reshape(Forward,5*PeriodSum,1);
mid_usedStandProb = 1/35;
f = figure('visible','off');
    mid_used01 = histogram(mid_used);
mid_used02 = mid_used01.Values;
mid_used02Prob = mid_used02/sum(mid_used02);  
[mid_used03,midused04] = sort(mid_used02);  
mid_used03Prob = mid_used03/sum(mid_used03); 
IndexPrior = find(mid_used03Prob>mid_usedStandProb, 1 );
ProbabilityPriorLabel = midused04(1,IndexPrior:end); 
ProbabilityPosteriorLabel = mid_nnn(end,1:5);  
MAPPosteriorLabel = intersect(ProbabilityPosteriorLabel,ProbabilityPriorLabel);
NumMap = length(MAPPosteriorLabel );
if NumMap == 0 
    SuggestedLabel = midused04(1,end-4:end); 
elseif NumMap > 0
    SuggestedLabel01 = setdiff(ProbabilityPriorLabel, MAPPosteriorLabel, 'stable'); 
    SuggestedLabel = [MAPPosteriorLabel, SuggestedLabel01(1,end-(5-NumMap)+1:end)]; % Before adapt
%     disp([SuggestedLabel,nnn(end,6:7)]);
end
if nnn_count < Summary-1 
    
%            [~,a] = sort(ForwardRW);
            [~,b] = sort(BackwardRW);
%            forecasting_series(i+1,:)= [a(end-4:end),b(end-1:end)];
    forecasting_series(nnn_count,:) = [SuggestedLabel,b(end-1:end)]; 
%     forecasting_series(nnn_count,:) = [SuggestedLabel,mid_nnn(end,6:7)];
    testing_series(nnn_count,:) = [nnn(nnn_count+1,1:5),nnn(nnn_count+1,6:7)];
    error(nnn_count,:) = [length(intersect(forecasting_series(nnn_count,1:5),testing_series(nnn_count,1:5))),length(intersect(forecasting_series(nnn_count,6:7),testing_series(nnn_count,6:7)))];
            
    
           if error(nnn_count,1) == 0 && error(nnn_count,2) == 0
             earnings(nnn_count,:) = 0; 
           elseif error(nnn_count,1) == 0 && error(nnn_count,2) == 1
             earnings(nnn_count,:) = 0; 
           elseif error(nnn_count,1) == 0 && error(nnn_count,2) == 2
             earnings(nnn_count,:) = 5; 
           elseif error(nnn_count,1) == 1 && error(nnn_count,2) == 0
             earnings(nnn_count,:) = 0;  
           elseif error(nnn_count,1) == 1 && error(nnn_count,2) == 1
             earnings(nnn_count,:) = 0; 
           elseif error(nnn_count,1) == 1 && error(nnn_count,2) == 2
             earnings(nnn_count,:) = 5; 
           elseif error(nnn_count,1) == 2 && error(nnn_count,2) == 0
             earnings(nnn_count,:) = 0; 
           elseif error(nnn_count,1) == 2 && error(nnn_count,2) == 1
             earnings(nnn_count,:) = 5; 
           elseif error(nnn_count,1) == 2 && error(nnn_count,2) == 2
             earnings(nnn_count,:) = 15; 
           elseif error(nnn_count,1) == 3 && error(nnn_count,2) == 0
             earnings(nnn_count,:) = 5;
           elseif error(nnn_count,1) == 3 && error(nnn_count,2) == 1
             earnings(nnn_count,:) = 15; 
           elseif error(nnn_count,1) == 3 && error(nnn_count,2) == 2
             earnings(nnn_count,:) = 200;             
           elseif error(nnn_count,1) == 4 && error(nnn_count,2) == 0
             earnings(nnn_count,:) = 100; 
           elseif error(nnn_count,1) == 4 && error(nnn_count,2) == 1
             earnings(nnn_count,:) = 300; 
           elseif error(nnn_count,1) == 4 && error(nnn_count,2) == 2
             earnings(nnn_count,:) = 3000;   
           elseif error(nnn_count,1) == 5 && error(nnn_count,2) == 0
             earnings(nnn_count,:) = 10000;
           elseif error(nnn_count,1) == 5 && error(nnn_count,2) == 1
             earnings(nnn_count,:) = 378250; 
           elseif error(nnn_count,1) == 5 && error(nnn_count,2) == 2
             earnings(nnn_count,:) = 10000000; 
           end
end
nnn_count= nnn_count+1;
if nnn_count > Summary
    disp(sum(error)/(Summary-2))
    disp([SuggestedLabel,nnn(end,6:7)]);
    disp((sum(earnings)-(Summary-2)*2)/(Summary-2));
    CountFlag=0;
end
end
    




