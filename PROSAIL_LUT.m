clc;clear;

data = load('Mead_Landsat.mat').output;
LUT = load('PROSAIL_uniform.mat').LUT;

LAIs = LUT(:,1);
LAI_pre = zeros(length(data),1);
for i = 1:length(data)
    [~,I] = sort(mean((LUT(:,2:7) - data(i,1:6)).^2,2));
    LAI_pre(i) = mean(LAIs(I(1:20)));
end

LAI_pre = reshape(LAI_pre,33,[])';
writematrix(LAI_pre, 'PROSAIL_LAI.txt');

plot(1:33,LAI_pre(1,:));