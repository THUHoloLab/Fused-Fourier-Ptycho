
clear
clc

reset(gpuDevice());
addpath(genpath('func_ware'));
addpath(genpath('cuFPM'));

img_all = 361;
% 使用鼠标，在要裁剪的图像部分上绘制矩形。
% 通过双击裁剪矩形内部或在上下文菜单中选择裁剪图像来执行裁剪操作

max_inten = 2^16-1;

color_full = {'red','green','blue'};

pix = 2048;
rect = [1,1];

img_rgb_final = [];
for color_index = 2

init_environment_rgb;
kc = [-M/2,-1+M/2,-N/2,-1+N/2];

imRaw_num = zeros(pix,pix,img_all,'single');
for led_num = 1:img_all
    i=ord_isum(led_num);
    j=ord_jsum(led_num);
        
    uo(led_num) = ledpos_true(i,j,1);
    vo(led_num) = ledpos_true(i,j,2);
    
    name = sprintf("%03d",k(i,j))
    imgname = "raw_images\" + color_full{color_index} + "\" + name + ".tif";

    temp = imread(imgname,'PixelRegion',{[rect(2),rect(2)+pix-1],...
                                         [rect(1),rect(1)+pix-1]});

    imRaw_num(:,:,led_num) = gpuArray(mean(single(temp),3));
end

imRaw_num = imRaw_num - min(imRaw_num(:));
imRaw_num = imRaw_num / max(imRaw_num(:));
imRaw_num = sqrt(imRaw_num);

foo = @(x) complex(gpuArray(single(x)));
wavefront1 = foo(((imresize(mean(imRaw_num(:,:,1),3),pratio)))); 
wavefront2 = foo(Pupil0);         

% imRaw_num = gpuArray(single(imRaw_num));

kt = kc(1) + vo;
kl = kc(3) + uo;
ledIdx = int32(gpuArray([kl;kt]));

tic
[wavefront1,wavefront2] = solve_FPM( ...
                "target",       wavefront1,...
                "pupil",        wavefront2,...
                "obsY",         imRaw_num,...
                "ledidx",       ledIdx,...
                "circ",         Pupil0,...
                "batchSize",    26);

wait(gpuDevice());
disp("solving FPM for " + toc + " s")

end

% support function for MATLAB - CUDA interplay
function [w1,w2] = solve_FPM(parms)

arguments
    parms.target;
    parms.pupil;
    parms.obsY;
    parms.ledidx;
    parms.circ;

    parms.pratio = 8;
    parms.epoch = 20;
    parms.batchSize = 26;
    parms.learningRate = 0.0005;
    parms.beta = 0.999;
    parms.eps = 1e-4;
end

parms.circ = gpuArray(single(parms.circ));
optimizer_arg = [parms.learningRate, parms.beta, parms.eps];

% launch CUDA function
[w1,w2] = cuFPM_pure(parms.target, ...
                     parms.pupil,...
                     parms.obsY, ...
                     parms.ledidx, ...
                     parms.circ,...
                     parms.pratio,...
                     parms.epoch,...
                     parms.batchSize,...
                     optimizer_arg);
end