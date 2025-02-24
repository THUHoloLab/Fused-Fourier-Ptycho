%% Efficient subpixel image registration by cross-correlation. 
% Registers two images (2-D rigid translation) within a  fraction 
% of a pixel specified by the user. Instead of computing a zero-padded FFT 
% (fast Fourier transform), this code uses selective upsampling by a
% matrix-multiply DFT (discrete FT) to dramatically reduce computation time and memory
% without sacrificing accuracy. With this procedure all the image points are used to
% compute the upsampled cross-correlation in a very small neighborhood around its peak. This 
% algorithm is referred to as the single-step DFT algorithm in [1].
%
% [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
% "Efficient subpixel image registration algorithms," Opt. Lett. 33, 
% 156-158 (2008).
%
% ----------------------------------------------------------------------- 
%
% Copyright (c) 2016, Manuel Guizar Sicairos, James R. Fienup, University of Rochester
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the University of Rochester nor the names
%       of its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
% --------------------------------------------------------------------------

%% Syntax
% The code receives the FFT of the reference and the shifted images, and an
% (integer) upsampling factor. The code expects FFTs with DC in (1,1) so do not use
% fftshift.
%
%    output = dftregistration(fft2(f),fft2(g),usfac);
%
% The images are registered to within 1/usfac of a pixel.
%
% output(1) is the normalized root-mean-squared error (NRMSE) [1] between f and
% g. 
%
% output(2) is the global phase difference between the two images (should be
% zero if images are real-valued and non-negative).
%
% output(3) and output(4) are the row and column shifts between f and g respectively. 
%
%    [output Greg] = dftregistration(fft2(f),fft2(g),usfac);
%
% Greg is an optional output, it returns the Fourier transform of the registered version of g,
% where the global phase difference [output(2)] is also compensated.


%% Obtain a reference and shifted images
% To illustrate the use of the algorithm, lets obtain a reference and a
% shifted image. First we read the reference image f(x,y)
clc
clear


path = 'imput\';

name = 'rawcd_rgb_large';

f = double(imread([path,name,'.png']))/255;

f_r = f(:,:,1);
f_b = f(:,:,3);

usfac = 100;
[~, Greg_r] = dftregistration(fft2(f(:,:,2)),fft2(f_r),usfac);
[~, Greg_b] = dftregistration(fft2(f(:,:,2)),fft2(f_b),usfac);

f(:,:,1) = it(abs(ifft2(Greg_r)),f(:,:,1));
f(:,:,3) = it(abs(ifft2(Greg_b)),f(:,:,3));

imwrite(f, ['imput/', name, num2str(2), '_', num2str(2), '.png']);



