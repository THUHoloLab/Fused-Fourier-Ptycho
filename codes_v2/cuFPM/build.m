clc
clear

disp("Building MATLAB extension for cuFPM");

forced_compiling_all = true;

tic

nvcc_flags = [...
    '-std=c++17 ',...
    '-allow-unsupported-compiler ' ...
];

setenv("NVCC_APPEND_FLAGS", nvcc_flags)

include_dirs = {...
    '', ...
    'cuda'};

flags = cellfun(@(dir) ['-I"' fullfile(pwd, dir) '"'], ...
                        include_dirs, 'UniformOutput', false); 
% use the cuda.lib
flags = [flags,{'-lcuda'},{'-lcufft'}]; 


cu_path  = 'cuda/';

cu_sources = {...
    'helpers.cu',...
    'kernel_foo.cu'};


main_file = 'cuFPM_pure.cu';

obj_path = fullfile(pwd, 'mex_obj/');
if ~exist(obj_path, 'dir')
    mkdir(obj_path);
end

% compiling for cuda file
cu_objs = cellfun(@(f) [obj_path replace(f,{'.cu'},{'.obj'})], ...
                        cu_sources, 'UniformOutput',false);
for i = 1:length(cu_sources)
    if ~exist(cu_objs{i},'file') || forced_compiling_all
        mexcuda(flags{:}, '-c', [cu_path,cu_sources{i}], ...
                          '-outdir', obj_path);
    else
        disp([cu_objs{i}, ' already exist, skip its compiling.']);
    end
end



mexcuda(flags{:}, main_file, cu_objs{:});

time_spend = toc;
disp(['compiling takes:',num2str(time_spend),'s'])