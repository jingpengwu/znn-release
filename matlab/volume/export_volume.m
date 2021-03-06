function [] = export_volume( fname, volume, ext )
% 
% Export 3D volume in binary format
% 
% Usage:
% 	export_volume( fname, volume )
% 	export_volume( fname, volume, ext )
% 	
% 	fname:	file name
% 	volume:	3D volume
%	ext: 	if exists, file name becomes [fname.ext]
%
% Results:
%	[fname]	or [fname.ext]	3D volume in binary format
%	[fname.size]			3D volume dimension in binary format
%
% Program written by:
% Kisuk Lee <kiskulee@mit.edu>, 2014

	% volume dimension
	fsz = fopen([fname '.size'], 'w');
	sz  = size(volume);	
	if ndims(sz) < 3
		sz = [sz 1];
	end
	fwrite(fsz, uint32(sz), 'uint32');

	% volume
	if exist('ext','var')		
		fvol = fopen([fname '.' ext], 'w');
	else
		fvol = fopen(fname, 'w');
	end		
	fwrite(fvol, double(volume), 'double');

end