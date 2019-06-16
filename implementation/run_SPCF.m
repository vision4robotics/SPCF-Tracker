function results = run_SPCF(seq)

params.hog_cell_size = 4;
params.fixed_area = 200^2;                     % standard area to which we resize the target
params.n_bins = 2^5;                           % number of bins for the color histograms (bg and fg models)
params.learning_rate_pwp = 0.01;               % bg and fg color models learning rate
params.lambda_scale = 0.1;                     % regularization weight

params.scale_lambda = 0.1;
params.scale_sigma_factor = 1/16;
params.scale_sigma = 0.1;
params.merge_factor = 0.3;

% fixed setup
params.hog_scale_cell_size = 4;                % Default DSST=4
params.scale_model_factor = 1.0;

params.feature_type = 'fhog';
params.scale_adaptation = true;
params.grayscale_sequence = false;	          % suppose that sequence is colour
params.merge_method = 'const_factor';


params.img_files = seq.s_frames;
params.img_path = '';

params.visualization = 1;
params.visualization_dbg = 1;

s_frames = seq.s_frames;
params.s_frames = s_frames;
params.video_path = seq.video_path;
im = imread([s_frames{1}]);

if(size(im,3)==1)
    params.grayscale_sequence = true;
end

region = seq.init_rect;

if(numel(region)==8)
    [cx, cy, w, h] = getAxisAlignedBB(region);
else
    x = region(1);
    y = region(2);
    w = region(3);
    h = region(4);
    cx = x+w/2;
    cy = y+h/2;
end

% init_pos is the centre of the initial bounding box
params.init_pos = [cy cx];
params.target_sz = round([h w]);
params.inner_padding = 0.2;                    % defines inner area used to sample colors from the foreground

[params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);

% in runTracker we do not output anything because it is just for debug
if params.visualization_dbg == 1
    params.fout = 1;
else
    params.fout = 0;
end

% HOG feature parameters
hog_params.nDim  = 31;
%   CN feature parameters
cn_params.nDim = 11;
%   Gray feature parameters
gray_params.nDim = 1;
%   Saliency feature parameters
saliency_params.nDim = 3;
%   Deep feature parameters
params.indLayers = [37, 28, 19];%   The CNN layers Conv3-4 in VGG Net
deep_params.nDim = [512, 512, 256];
deep_params.layers = params.indLayers;
%   handcrafted parameters
Feat1 = 'conv3'; % fhog, cn, gray, saliency, handcrafted_assem
Feat2 = 'conv5'; % deep_assem, conv3, conv4, conv5
switch Feat1
    case 'conv3'
        params.layerInd{1} = 3;
        params.nDim{1} = 256;
    case 'conv4'
        params.layerInd{1} = 2;
        params.nDim{1} = 512;
    case 'conv5'
        params.layerInd{1} = 1;
        params.nDim{1} = 512;
    case 'fhog'
        params.layerInd{1} = 0;
        params.nDim{1} = 31;
    case 'cn'
        params.layerInd{1} = 0;
        params.nDim{1} = 11;
    otherwise
        params.layerInd{1} = 0;
end

switch Feat2
    case 'conv3'
        params.layerInd{2} = 3;
        params.nDim{2} = 256;
    case 'conv4'
        params.layerInd{2} = 2;
        params.nDim{2} = 512;
    case 'conv5'
        params.layerInd{2} = 1;
        params.nDim{2} = 512;
    case 'fhog'
        params.layerInd{2} = 0;
        params.nDim{2} = 31;
    case 'cn'
        params.layerInd{2} = 0;
        params.nDim{2} = 11;
    otherwise
        params.layerInd{2} = 0;
end
params.feat_type = {Feat1, Feat2};

params.t_global.type_assem = 'fhog_cn'; % fhog_cn, fhog_gray,fhog_cn_gray_saliency, fhog_gray_saliency,fhog_cn_gray,fhog_gray
switch params.t_global.type_assem
    case 'fhog_cn_gray_saliency'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim + gray_params.nDim + saliency_params.nDim;
    case 'fhog_cn_gray'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim + gray_params.nDim;
    case 'fhog_gray_saliency'
        handcrafted_params.nDim = hog_params.nDim + gray_params.nDim + saliency_params.nDim;
    case 'fhog_gray'
        handcrafted_params.nDim = hog_params.nDim + gray_params.nDim;
    case 'fhog_cn'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim;
end

params.t_features = {struct('getFeature_fhog',@get_fhog,...
    'getFeature_cn',@get_cn,...
    'getFeature_gray',@get_gray,...
    'getFeature_saliency',@get_saliency,...
    'getFeature_deep',@get_deep,...
    'getFeature_handcrafted',@get_handcrafted,...
    'hog_params',hog_params,...
    'cn_params',cn_params,...
    'gray_params',gray_params,...
    'saliency_params',saliency_params,...
    'deep_params',deep_params,...
    'handcrafted_params',handcrafted_params)};

params.t_global.w2c_mat = load('w2c.mat');
params.t_global.factor = 0.2; % for saliency
params.t_global.cell_size = 4;
params.t_global.cell_selection_thresh = 0.75^2;

params.lambda1 = 0.001;
params.lambda2 = 0.0001;
params.gamma = 0.0001;                       %repress the anomalies in different feature response
params.output_sigma_factor = {1/8, 1/8};
params.kernel_type = 'linear'; % 'gaussian' 'polynomial' 'linear'
params.tran_sigma = {0.5, 0.5};
params.polya = {1,1};
params.polyb = {3,3};

tran_lr{1} = 0.01;
tran_lr{2} = 0.03;
params.learning_rate_cf = tran_lr;

params.num_scales = 33;
params.scale_step = 1.02;
params.scale_model_max_area = 32*16;
params.learning_rate_scale = 0.01;



%   Sample purification
params.nSamples = 50;
params.sample_reg = {5, 2};                % Weights regularization (mu)
params.sample_burnin = 10;              % Number of frames before weight optimization starts
params.num_acs_iter = 1;                % Number of Alternate Convex Search iterations
params.sample_replace_strategy = 'constant_tail';
params.lt_size = 10;

%	start the actual tracking
results = tracker(params, im, bg_area, fg_area, area_resize_factor);