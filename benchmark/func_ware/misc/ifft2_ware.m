function y = ifft2_ware(x,if_shift)

if if_shift
    y =  ifft2(ifftshift(ifftshift(x,2),1));
else
    y = ifft(ifft(x, [], 2), [], 1, 'nonsymmetric');
end

end