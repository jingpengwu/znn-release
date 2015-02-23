%% prepare the dataset for ZNN
% Program written by:
% Jingpeng Wu <jingpeng@princeton.edu>, 2014
run ../matlab/addpath_recurse

%% parameters
% the number of batches that need to generate
BatchNum = 1;

%% the training dateset

vol_train = loadtiff('fish/data/original/RawInput_8bit_Daan_train.tif');
vol_label = loadtiff('fish/data/original/ExportBoundaries_8bit_Daan.tif');

% %% cut volume
% vol_train = vol_train(end-255:end, end-255:end,:);
% vol_label = vol_label(end-255:end, end-255:end,:);

%% divide the volume to small volumes
vol_size = size(vol_train);
sub_size = [504 504 127];
batch_id = 0;
for m = 1 : ceil(vol_size(1)/sub_size(1))
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
            export_volume(['fish/data/batch' num2str(batch_id)], subvol_train, 'image');
            
            subvol_label = vol_label(m1:m2, n1:n2, k1:k2);
            export_volume(['fish/data/batch' num2str(batch_id)], subvol_label, 'label');
            
            % size of subvolumes
            sz = size(subvol_train);
            % generate the spec files with volume size
            fname = ['fish/spec/batch' num2str(batch_id) '.spec'];
            delete(fname);  fspec = fopen(fname, 'w');
            fprintf(fspec, ['[INPUT1]\npath=./dataset/fish/data/batch' num2str(batch_id) '\n']);
            fprintf(fspec, ['ext=image\nsize=' num2str(sz(1)) ','...
                num2str(sz(2)) ',' num2str(sz(3)) '\npptype=standard2D\n\n']);
            fprintf(fspec, ['[LABEL1]\npath=./dataset/fish/data/batch' num2str(batch_id) '\n']);
            fprintf(fspec, ['ext=label\nsize=' num2str(sz(1)) ',' num2str(sz(2)) ',' ...
                num2str(sz(3)) '\npptype=binary_class\n\n']);
            
            fprintf(fspec, ['[MASK1]\nsize=' num2str(sub_size(1)) ',' num2str(sz(2)) ',' ...
                num2str(sz(3)) '\npptype=one\nppargs=2']);
            fclose(fspec);
        end
    end
end