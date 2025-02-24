
clear
clc

reset(gpuDevice());
addpath(genpath('func_ware'));
addpath(genpath('fusedfpm'));
img_all = 361;
% 使用鼠标，在要裁剪的图像部分上绘制矩形。
% 通过双击裁剪矩形内部或在上下文菜单中选择裁剪图像来执行裁剪操作

max_inten = 2^16-1;

color_full = {'red','green_test','blue'};

img_center = imread(['raw_images\',color_full{2},'\181.tif']);
[temp,rect] = imcrop(img_center);
if rem(size(temp,1),2) == 1
    rect(4) = rect(4) - 1;
end
if rem(size(temp,2),2) == 1
    rect(3) = rect(3) - 1;
end
pix = fix((rect(4) + rect(3))/2);
pix = pix + mod(pix,2);
pix = 512;
rect = fix(rect);
close all

% pix = 2048;
% rect = [1,1];
% load loc_pos.mat;

img_rgb_final = [];
for color_index = 2
imRaw = zeros(pix,pix,img_all);
Ibk = zeros(1,img_all);

%% Load data, crop image
for num_of_image = 1:img_all
    clc
    disp(num_of_image);

    name = sprintf("%03d",num_of_image);

    imgname = "raw_images\" + color_full{color_index} + "\" +...
                        name + ".tif";
        
    temp = imread(imgname,'PixelRegion',{[rect(2),rect(2)+pix-1],...
                                         [rect(1),rect(1)+pix-1]});
    imRaw(:,:,num_of_image) = single(temp);
end

clc

imRaw = imRaw - min(imRaw(:));
imRaw = sqrt(imRaw / max(imRaw(:)));

init_environment_rgb;
kc = [-M/2,-1+M/2,-N/2,-1+N/2];
imRaw_num = imRaw;
%% preparing reconstruction data for WARE engine
% imRaw_new = imRaw;
for led_num = 1:img_all
    i=ord_isum(led_num);
    j=ord_jsum(led_num);
        
    uo(led_num) = ledpos_true(i,j,1);
    vo(led_num) = ledpos_true(i,j,2);

    imRaw_num(:,:,led_num) = imRaw(:,:,k(i,j));
end

led_pos  = [vo',uo'];

batchSize = 16; 

numEpochs = 20;
numIterationsPerEpoch  = img_all / batchSize;
numIterations = numEpochs * numIterationsPerEpoch;


%% parameters for optimizers
learning_rate = 0.002;
optimizer_w1 = optimizer_RMSprop(0,0,0.999,0,false,learning_rate);
optimizer_w2 = optimizer_RMSprop(0,0,0.999,0,false,learning_rate);

%% The iterative recovery process for FP
intensity_constrain = 0.75;      % sample's intensity constrain
amplitude_constrain = 0.1;       % pupil function's amplitude constrain

foo = @(x) complex(gpuArray(single(x)));

wavefront1 = foo(imresize(mean(imRaw_num(:,:,1:9),3),pratio)); 
wavefront2 = foo(Pupil0);                             

imRaw_num = gpuArray(single(imRaw_num));
led_pos = gpuArray(single(led_pos));

epoch = 0;
iteration = 0;
type = 'none';   

clear imRaw uo vo;
c = 0;
while epoch < numEpochs
    epoch = epoch + 1;
    
    first = 1;
    last = 1;
    while last < img_all
        iteration = iteration + 1;
        disp("epoch: " + epoch + "--" + iteration);
        last = first + batchSize - 1;
        leds = led_pos(first:min(last,img_all),:);
      

        %% forward propagation, gain gradient
        dldw1 = 0;
        dldw2 = 0;
        
        %{ 
        % this is conventional GPU accelerated FPM 
        [loss,dldw1,dldw2] = fpm_forward_GPU(wavefront1, wavefront2 , ...
                                                     kc, ...
                                                     leds, ...
                                                     imRaw_num(:,:,first:min(last,img_all)), ...
                                                     pratio, ...
                                                     denoise,type);
        %}

        % this is the CUDA implementarion of FPM
        kt = kc(1) + leds(:,1);
        kl = kc(3) + leds(:,2);
        ledIdx = int32(gpuArray([kl';kt']));
        [dldw1,dldw2] =  fullyfusedFPM(wavefront1, ...
                                       wavefront2,...
                                       imRaw_num(:,:,first:min(last,img_all)), ...
                                       ledIdx,pratio);
        wait(gpuDevice());
    
        first = last + 1;
        
        %% learning the parameters

        wavefront1 = optimizer_w1.step(wavefront1,dldw1);
        wavefront2 = optimizer_w2.step(wavefront2,dldw2);

        wavefront2 = max(min(abs(wavefront2),...
                                Pupil0 + amplitude_constrain),...
                                Pupil0 - amplitude_constrain)...
                                        .* sign(wavefront2) .* Pupil0;
 
        % clc
        % disp(['processing :',fix(num2str(100 * epoch/numEpochs)*100)/100,' %']);
    end
% 
    if mod(epoch,1) == 0
        o = (wavefront1 + c);%rot90(wavefront1,2);
        figure(5);
        subplot(1,2,1)
        pinpu=log(abs(fftshift(fft2(o)))+1);
        imshow(angle(wavefront2),[]);
        title('Fourier spectrum');

        % Show the reconstructed amplitude
        subplot(1,2,2)
        imshow(abs(o).^2,[]);
        drawnow;
    end
%     denoise(1) = denoise(1) .* 1.1;
end

imwrite(mat2gray(abs(wavefront1).^2),[color_full{color_index},'_out.png'])
imwrite(mat2gray(imRaw_num(:,:,1).^2),[color_full{color_index},'_raw.png'])

end



