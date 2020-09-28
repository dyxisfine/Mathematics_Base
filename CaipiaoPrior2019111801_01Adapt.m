close all;clear;clc;
% load('ticai.mat')
% load('ticai01.mat')
load('ticai20004.mat') %只考虑此前一个时间点
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
       % 加入RNN的后区计算
        FowwardRW = 1./(1+exp(-ones(1,35)*(1/35)));
        BackwardRW = 1./(1+exp(-ones(1,12)*(1/12)));
while CountFlag>0
         BiasF =  nnn(nnn_count,1:5);
         BiasB =  nnn(nnn_count,6:7);
         [ Bias_usedF ] = SampleExpanding( BiasF, 35);
         [ Bias_usedB ] = SampleExpanding( BiasB, 12);
             Current_Bias = [1./(1+exp(-Bias_usedF)), 1./(1+exp(-Bias_usedB))];
             output_layer = ([FowwardRW,BackwardRW]) + 1.8 * Current_Bias;
             % 更新先验信息
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

P_1_1 = length(find (Record ==1))/(PeriodSum-1); %有一个前区数字重合的概率
P_2_1 = length(find (Record ==2))/(PeriodSum-1); %有两个前区数字重合的概率
P_3_1 = length(find (Record ==3))/(PeriodSum-1); %有三个前区数字重合的概率
% nchoosek(n,m)
P_1Hat =( nchoosek(35,4)* nchoosek(5,1) * nchoosek(34,4) )/( nchoosek(35,5)^(2) );%无偏情况下有一个前区数字重合的概率 {C2,12*C1,2*C1,10}/{C2,12*C2*12}
P_2Hat = ( nchoosek(35,3)* nchoosek(5,2) * nchoosek(33,3) )/( nchoosek(35,5)^(2) );%无偏情况下有两个前区数字重合的概率 {C2,12*C1,2*C1,10}/{C2,12*C2*12}
P_3Hat = ( nchoosek(35,2)* nchoosek(5,3) * nchoosek(32,2) )/( nchoosek(35,5)^(2) );%无偏情况下有三个前区数字重合的概率 {C2,12*C1,2*C1,10}/{C2,12*C2*12}

%实验结果证明前区号码并不是完美的无偏估计，所以上一期前区号码中有一个数字在本区会以概率P_1_1
%出现，本期号码的其余数字应该按照出现概率轮流选择，并与P_2_1,P_3_1 对比


% 计算到现在为止，前区号码出现概率
mid_used = reshape(Forward,5*PeriodSum,1);
mid_usedStandProb = 1/35;
f = figure('visible','off');
    mid_used01 = histogram(mid_used);
mid_used02 = mid_used01.Values;
mid_used02Prob = mid_used02/sum(mid_used02);  % 已出现的数字的频数表
% 已经发生的事情就是概率发生最大的事情___贝叶斯学派--针对不公平的实验
% 已经发生的事情不影响下一步将要发生的事情___频率学派--针对公平的实验
[mid_used03,midused04] = sort(mid_used02);  % mid_used04 接下来出现概率从小到大排列的的数字表
mid_used03Prob = mid_used03/sum(mid_used03); % mid_used04 接下来出现概率表
% 检查先验概率表中的大概率事件
IndexPrior = find(mid_used03Prob>mid_usedStandProb, 1 );
ProbabilityPriorLabel = midused04(1,IndexPrior:end); % 先验概率表中的大概率事件
ProbabilityPosteriorLabel = mid_nnn(end,1:5);  % 后验概率数字直接定义为上一期出现的数字
% 一个上一期出现了的数字，每一个都有P_1_1=43%的概率在下一期出现，连续两期号码一样的概率是 1%， 
% 且确实发生过 http://sports.sina.com.cn/l/2010-03-31/17234914939.shtml


MAPPosteriorLabel = intersect(ProbabilityPosteriorLabel,ProbabilityPriorLabel);
NumMap = length(MAPPosteriorLabel );
if NumMap == 0 % 先验没有和后验重合的数字
    % 此时先验数字在下一次实验出现的概率都是P_1_1, which is much higher than standard.
    % 此外每一个数字还有它自身出现的概率，需要做出一个数字出现后验概率表，对于19131样本，先验小概率事件发生两次，与后验概率事件无关
    % 如果没有数字出现在MAP中，那么下一期数字与本期的概率相同的概率在1%，远超通过先验概率表正确预测5个数字的概率
%     SuggestedLabel = ProbabilityPosteriorLabel;  % Before adapt 目前为止成功率最高的情况
    SuggestedLabel = midused04(1,end-4:end); % 测试20200108   % Prior
%     disp([SuggestedLabel,nnn(end,6:7)]);
elseif NumMap > 0 % 先验有超过一个和后验重合的数字
    SuggestedLabel01 = setdiff(ProbabilityPriorLabel, MAPPosteriorLabel, 'stable'); % 求差集
    SuggestedLabel = [MAPPosteriorLabel, SuggestedLabel01(1,end-(5-NumMap)+1:end)]; % Before adapt
%     disp([SuggestedLabel,nnn(end,6:7)]);
end
if nnn_count < Summary-1 % 为了保证有test序列存在
    
%            [~,a] = sort(ForwardRW);
            [~,b] = sort(BackwardRW);
%            forecasting_series(i+1,:)= [a(end-4:end),b(end-1:end)];
    forecasting_series(nnn_count,:) = [SuggestedLabel,b(end-1:end)]; % 前区来自贝叶斯推断，后区来自RNN神经网络
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
    




