%% prepare the dataset for ZNN
% Program written by:
% Jingpeng Wu <jingpeng@princeton.edu>, 2014
clc
clear
run ../addpath_recurse

%% parameters
% the maximum number of batches that need to generate
BatchNum = 200;

% the training dateset
Dir = '../../dataset/fish/';
vol_train = loadtiff([Dir 'data/original/Daan_raw.tif']);
vol_label = loadtiff([Dir 'data/original/Daan_label.tif']);

% vol_train = loadtiff([Dir 'data/original/Kyle_raw.tif']);
% vol_label = loadtiff([Dir 'data/original/Kyle_label.tif']);
% % % 
% vol_train = loadtiff([Dir 'data/original/Merlin_raw2.tif']);
% vol_label = loadtiff([Dir 'data/original/Merlin_label2.tif']);

% vol_train = loadtiff([Dir 'data/original/raw.tif']);

% the beginning batch ID
batch_id = 24-1;


%% shrink volume
sz = size( vol_train );
new_sz = floor(sz/2);
new_sz(3) = sz(3);
new_vol_train = zeros(new_sz, 'uint8');
new_vol_label = zeros(new_sz, 'uint8');
for k = 1 : sz(3)
    new_vol_train(:,:,k) = imresize(vol_train(:,:,k), new_sz(1:2));
    new_vol_label(:,:,k) = imresize(vol_label(:,:,k), new_sz(1:2), 'nearest');
end

vol_train = new_vol_train;
vol_label = new_vol_label;


%% divide the volume to small volumes
vol_size = size(vol_train);
sub_size = [6000 6000 5000];

for m = 1 : ceil(vol_size(1)/sub_size(1))
    m
    m1 = (m-1)*sub_size(1) +1;
    m2 = min(m*sub_size(1), vol_size(1));
    for n = 1 : ceil(vol_size(2)/sub_size(2))
        n1 = (n-1)*sub_size(2) +1;
        n2 = min(n*sub_size(2), vol_size(2));
        for k = 1 : ceil(vol_size(3) / sub_size(3))
            k1 = (k-1)*sub_size(3) +1;
            k2 = min(k*sub_size(3), vol_size(3));
            
            % export the batch images
            batch_id = batch_id + 1;
            
            if batch_id > BatchNum
                break;
            end
            
            subvol_train = vol_train(m1:m2, n1:n2, k1:k2);
            export_volume([Dir 'data/batch' num2str(batch_id)], subvol_train, 'image');
            
            subvol_label = vol_label(m1:m2, n1:n2, k1:k2);
            export_volume([Dir 'data/batch' num2str(batch_id)], subvol_label, 'label');
            
            % size of subvolumes
            sz = size(subvol_train);
            % generate the spec files with volume size
            fname = [Dir 'spec/batch' num2str(batch_id) '.spec'];
            fname
            delete(fname);  fspec = fopen(fname, 'w');
            fprintf(fspec, ['[INPUT1]\npath=./dataset/fish/data/batch' num2str(batch_id) '\n']);
            fprintf(fspec, ['ext=image\nsize=' num2str(sz(1)) ','...
                num2str(sz(2)) ',' num2str(sz(3)) '\npptype=standard2D\n\n']);
            fprintf(fspec, ['[LABEL1]\npath=./dataset/fish/data/batch' num2str(batch_id) '\n']);
            fprintf(fspec, ['ext=label\nsize=' num2str(sz(1)) ',' num2str(sz(2)) ',' ...
                num2str(sz(3)) '\npptype=binary_class\n\n']);
            
            fprintf(fspec, ['[MASK1]\nsize=' num2str(sz(1)) ',' num2str(sz(2)) ',' ...
                num2str(sz(3)) '\npptype=one\nppargs=2']);
            fclose(fspec);
        end
    end
end