% 13*13
%% Simulate the forward imaging process of Fourier ptychography
function cuFPM_recon_2025_circular(numEpochs,batchSize)
reset(gpuDevice());

addpath(genpath('func_ware'));
addpath(genpath('cuFPM'));

abs_path = 'E:\ShuheZhang@Tsinghua\papers\《光精》CUDA\Simulation\';

name_sequence = {'raw_data\'};


path = {'r/','g/','b/'};

led_num = [1,8,12,16,24,36,48];
led_total = sum(led_num(:));
rot_ang = 0 / 180 * pi;

% pix = 512;
% gpuDevice(1)
global pixel_super
pixel_super = 1;

% rect = [1024,1024];

name = 100;

learning_rate = [0.0015,0.0015,0.0015];

% 

crop_img = false;

for group = 1

    if crop_img
        figure;
        [temp,rect] = imcrop((imread([abs_path,name_sequence{group},path{2},'1.tif'])));
        if rem(size(temp,1),2) == 1
            rect(4) = rect(4) - 1;
        end
        if rem(size(temp,2),2) == 1
            rect(3) = rect(3) - 1;
        end
        pix = fix((rect(4) + rect(3))/2);
        pix = pix + mod(pix,2);
        rect = fix(rect);
        save("loc_pos.mat","pix","rect")
    else
        pix = 512;
        rect = [1,1];
    end

for color_index = 2

% load loc_pos.mat
imRaw_new = zeros(pix,pix,led_total,'single');


% Load data, crop image
for num_of_image = 1:led_total
    clc
    disp(num_of_image);
    img = single(imread([abs_path,name_sequence{group},path{color_index},'img',num2str(num_of_image),'.tif'], ...
                          'PixelRegion',{[rect(2),rect(2)+pix-1],...
                                         [rect(1),rect(1)+pix-1]}));
    clear img_full;
    imRaw_new(:,:,num_of_image) = gpuArray(mean(img,3));
end

imRaw_new = imRaw_new - min(imRaw_new(:));
imRaw_new = imRaw_new / max(imRaw_new(:));
% imRaw_new = sqrt(imRaw_new);
% imRaw_new = gpuArray(single(sqrt(imRaw_new)));

img_raw_rgb(:,:,color_index) = imRaw_new(:,:,1);


[f_pos_set_true,pratio,Pupil0] = init_environment_rgb(color_index,...
                                                      pix, ...
                                                      led_num, ...
                                                      rot_ang);

% (0~225) set mini-batch size a total of 225 images for FPM recon
% batchSize = 36; 

% numEpochs = 5;
numIterationsPerEpoch  = size(imRaw_new,3) / batchSize;
numIterations = numEpochs * numIterationsPerEpoch;

epoch = 0;
iteration = 0;

%% The iterative recovery process for FP
disp('initializing parameters')
foo = @(x) complex(gpuArray(single(x)));

oI = (imresize(mean(imRaw_new(:,:,1),3),pratio * pixel_super)); 
wavefront1 = foo(oI);
wavefront2 = foo(Pupil0);   


disp('begin solving-----')

error_bef = inf;

optimizer_w1 = optimizer_RMSprop(0,0,0.999,0,false,learning_rate(color_index));
optimizer_w2 = optimizer_RMSprop(0,0,0.999,0,false,learning_rate(color_index));

% optimizer_w1 = optimizer_yogi(0,0,0.9,0.999,learning_rate(color_index));
% optimizer_w2 = optimizer_yogi(0,0,0.9,0.999,learning_rate(color_index) * 2);

psnr_data = [];
ssim_data = [];
time_data = [];

gt = single(imread('source_image/raw_img.tif'))/255;
gt = gt(:,:,color_index);
ac = 0.1;
% kc = [-1023,1024,-1023,1023];

while epoch < numEpochs
    epoch = epoch + 1;

    first = 1;
    last = 1;

    this_timer = 0;
    while last < led_total
        iteration = iteration + 1;

        last = first + batchSize - 1;
        leds = f_pos_set_true(first:min(last,led_total),:);
        
        

        kt = leds(:,3);
        kl = leds(:,1);

        ledIdx = int32(gpuArray([kl';kt']));
        

        % kernel_foo = @() fullyfusedFPM(wavefront1, ...
        %                                wavefront2,...
        %                                imRaw_new(:,:,first:min(last,led_total)), ...
        %                                ledIdx,pratio);
        % tt = gputimeit(kernel_foo,2);
        % wait(gpuDevice());

        % cuda based FPM
        start = tic;
        [dldw1,dldw2] =  fullyfusedFPM(wavefront1, ...
                                       wavefront2,...
                                       imRaw_new(:,:,first:min(last,led_total)), ...
                                       ledIdx,pratio);
        
        % conventional GPU
        % [loss,dldw1,dldw2] = fpm_forward_GPU(wavefront1, ...
        %                                         wavefront2, ...
        %                                         0, ...
        %                                         leds, ...
        %                                         imRaw_new(:,:,first:min(last,led_total)), ...
        %                                         pratio, ...
        %                                         0.1,'none');

        wait(gpuDevice());
        % toc(start)
        this_timer = this_timer + toc(start);
    
        first = last + 1;
  
        % timer = timer + toc;
       
        %% learning the parameters
        % w2 = medfilt2(real(wavefront1),[3,3]) + 1i *medfilt2(imag(wavefront1),[3,3]);

        % dldw1 = dldw1 + 20* (wavefront1 - w2); 

        wavefront1 = optimizer_w1.step(wavefront1,dldw1);
        wavefront2 = optimizer_w2.step(wavefront2,dldw2);

        wavefront2 = max(min(abs(wavefront2),...
                                Pupil0 + ac),...
                                Pupil0 - ac)...
                                        .* sign(wavefront2) .* Pupil0;
    end
    
    time_data = [time_data,this_timer];
    w2 = gather(wavefront1);
    this_psnr = 0;
    this_ssim = 0;
    % this_psnr = psnr(gt,single((abs(w2))));
    % this_ssim = ssim(gt,single((abs(w2))));
    psnr_data = [psnr_data,this_psnr];
    ssim_data = [ssim_data,this_ssim];
    figure(111);
    subplot(121);plot(cumsum(time_data),psnr_data);
    subplot(122);plot(cumsum(time_data),ssim_data);
    drawnow;
    clc
    
    % if abs(error_bef - error_now)/error_bef < 0.06
    %     optimizer_w1.lr = optimizer_w1.lr * 0.5;
    %     optimizer_w2.lr = optimizer_w2.lr * 0.5;
    % end
    % error_bef = error_now;

    
    sprintf("at %d epoch, takes = %2f",epoch,this_timer)
    % if epoch > (numEpochs*0.6)
    optimizer_w1.lr = optimizer_w1.lr * 0.75;
    optimizer_w2.lr = optimizer_w2.lr * 0.75;
    % end

    % if mod(epoch,1) == 0 
    %     w2 = gather(fftshift(fft2(wavefront1)));
    %     figure(7);
    %     imshow(log(abs(w2).^2 + 1),[]);
    %     title(['Iteration No. = ',int2str(epoch), '  \alpha = ',num2str(optimizer_w1.lr)])
    %     drawnow;
    % end
%     denoise(1) = denoise(1) .* 1.1;
end
% mkdir([name_sequence{group}(1:8)])
% imwrite(uint16((2^16-1) * mat2gray(abs(wavefront1))),['test_out_c',num2str(color_index),'iter = 5.tif']);

save(['MATLAB_quality_time_',num2str(numEpochs),'iter_B=',num2str(batchSize),'_test.mat'],'ssim_data','psnr_data','time_data')
% img_rgb_final(:,:,color_index) = wavefront1; 

end
% save([name_sequence{group},'data_elfpie.mat'],'img_rgb_final','wavefront2','-v7.3')
end
end

% f = img_rgb_final;
% % f_r = f(:,:,1);
% % f_b = f(:,:,3);
% % usfac = 1500;
% 
% f(:,:,1) = 0.7*f(:,:,1);
% f(:,:,2) = 0.9*f(:,:,2);
% f(:,:,3) = 0.7*f(:,:,3);
% 
% % f = img_rgb_raw;
% f = gather(f);
% title('请选择背景区域')
% [temp,rect] = imcrop(f);
% if rem(size(temp,1),2) == 1
%     rect(4) = rect(4) - 1;
% end
% if rem(size(temp,2),2) == 1
%     rect(3) = rect(3) - 1;
% end
% pix = fix((rect(4) + rect(3))/2);
% pix = pix + mod(pix,2);
% rect = fix(rect);
% % close all
% 
% area = f(rect(2):rect(2)+pix-1,rect(1):rect(1)+pix-1,:);
% 
% r_sum = mean(mean(area(:,:,1)));
% g_sum = mean(mean(area(:,:,2)));
% b_sum = mean(mean(area(:,:,3)));
% 
% white = [210,210,210]/255;
% f_ic = f;
% 
% f_ic(:,:,1) = f(:,:,1) .* white(1) / r_sum;
% f_ic(:,:,2) = imfilter(f(:,:,2),fspecial('gaussian',5,1),'symmetric') .* white(2) / g_sum;
% f_ic(:,:,3) = imfilter(f(:,:,3),fspecial('gaussian',5,1),'symmetric') .* white(3) / b_sum;
% 
% figure();imshow(f_ic,[])
% 
% 
% % lab_img = rgb2lab(f_ic);
% % lab_img(:,:,1) = adapthisteq(lab_img(:,:,1)/100) * 100;
% % f_ic = 0.4*f_ic + 0.6*lab2rgb(lab_img);
% % save rgb_img f_ic
% imwrite(double(f_ic),'data_ware_20240918_raw2.tif');


