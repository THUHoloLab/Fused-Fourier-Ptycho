function y = fft2_ware(x,if_shift)

if if_shift
    y =  fftshift(fftshift(fft2(x),1),2);
else
    y = fft(fft(x,[],2),[],1);
end

end