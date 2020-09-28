close all;clear;clc;
% load('ticai.mat')
% load('ticai01.mat')
load('ticai20004.mat')
% 能达到每10期预测对8个前区与4个后区数字
rng(117, 'twister')  % 控制随机数
% PeriodNum =3;
% nnn = [zeros(1,14);nnn];
[PeriodNum, ~] = size(nnn);
return
ForwardR = nnn(:,1:5);
BackwardR = nnn (:,6:7);
% 
% a = Forward;
% Forward=a(end:-1:1,:);
%   更换顺序
% a = Backward;
% Backward=a(end:-1:1,:);



Matrix = -2:0.1:2;
for j = 1
FowwardRW = 1./(1+exp(-ones(1,35)*(1/35)));
BackwardRW = 1./(1+exp(-ones(1,12)*(1/12)));
    
alpha = 1.8;
for i=1:PeriodNum

    BiasF =  nnn(i,1:5);
    BiasB =  nnn(i,6:7);
        [ Bias_usedF ] = SampleExpanding( BiasF, 35);
        [ Bias_usedB ] = SampleExpanding( BiasB, 12);
    Current_Bias = [1./(1+exp(-Bias_usedF)), 1./(1+exp(-Bias_usedB))];
    % 先验后验置信度不能相等
%     output_layer = ([FowwardRW,BackwardRW]) + Current_Bias;
    output_layer = ([FowwardRW,BackwardRW]) + alpha * Current_Bias;
%     if i==1
%         return
%     end
% 更新先验信息
    ForwardRW =output_layer(1,1:35);
    BackwardRW = output_layer(1,36:47);
%         [ ForwardRW  ] = SampleExpanding( ForwardRW , 35);
%         [ BackwardRW ] = SampleExpanding( BackwardRW, 12);
       if i<PeriodNum
            [~,a] = sort(ForwardRW);
            [~,b] = sort(BackwardRW);
           forecasting_series(i+1,:)= [a(end-4:end),b(end-1:end)];
           testing_series(i+1,:) = [nnn(i+1,1:5), nnn(i+1,6:7)];
           error(i+1,:) = [length(intersect(forecasting_series(i+1,1:5),testing_series(i+1,1:5))),length(intersect(forecasting_series(i+1,6:7),testing_series(i+1,6:7)))];
           
           
           if error(i+1,1) == 0 && error(i+1,2) == 0
             earnings(i+1,:) = 0; 
           elseif error(i+1,1) == 0 && error(i+1,2) == 1
             earnings(i+1,:) = 0; 
           elseif error(i+1,1) == 0 && error(i+1,2) == 2
             earnings(i+1,:) = 5; 
           elseif error(i+1,1) == 1 && error(i+1,2) == 0
             earnings(i+1,:) = 0;  
           elseif error(i+1,1) == 1 && error(i+1,2) == 1
             earnings(i+1,:) = 0; 
           elseif error(i+1,1) == 1 && error(i+1,2) == 2
             earnings(i+1,:) = 5; 
           elseif error(i+1,1) == 2 && error(i+1,2) == 0
             earnings(i+1,:) = 0; 
           elseif error(i+1,1) == 2 && error(i+1,2) == 1
             earnings(i+1,:) = 5; 
           elseif error(i+1,1) == 2 && error(i+1,2) == 2
             earnings(i+1,:) = 15; 
           elseif error(i+1,1) == 3 && error(i+1,2) == 0
             earnings(i+1,:) = 5;
           elseif error(i+1,1) == 3 && error(i+1,2) == 1
             earnings(i+1,:) = 15; 
           elseif error(i+1,1) == 3 && error(i+1,2) == 2
             earnings(i+1,:) = 200;             
           elseif error(i+1,1) == 4 && error(i+1,2) == 0
             earnings(i+1,:) = 100; 
           elseif error(i+1,1) == 4 && error(i+1,2) == 1
             earnings(i+1,:) = 300; 
           elseif error(i+1,1) == 4 && error(i+1,2) == 2
             earnings(i+1,:) = 3000;   
           elseif error(i+1,1) == 5 && error(i+1,2) == 0
             earnings(i+1,:) = 10000;
           elseif error(i+1,1) == 5 && error(i+1,2) == 1
             earnings(i+1,:) = 378250; 
           elseif error(i+1,1) == 5 && error(i+1,2) == 2
             earnings(i+1,:) = 10000000; 
           end
           
       end
        
end

    Forward =output_layer(1,1:35);
    Backward = output_layer(1,36:47);
[~,a] = sort(Forward);
[~,b] = sort(Backward);
disp([a(end-4:end),b(end-1:end)]);
Summary = PeriodNum - 1;
    disp(sum(error)/Summary)
    disp((sum(earnings)-Summary*2)/Summary);
    ee(j) = (sum(earnings)-Summary*2)/Summary;
    
    
end
