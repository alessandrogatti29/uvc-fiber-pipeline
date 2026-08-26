function map_fibers_uvc(sourceVtuPath, targetVtuPath, sourceRPath, targetRPath, outputPath, cobivecoRoot, searchradius)
% Step 4: Map fiber/sheet/normal orientation from a source mesh onto a
% target mesh via UVC correspondence. Use this when source and target are
% DIFFERENT shapes (e.g. a generic template vs. a patient-specific cut
% biventricular mesh) -- there's no direct spatial correspondence between
% them, only a correspondence through Cobiveco's universal ventricular
% coordinates (ab/rt/tm/tv).
%
% Both meshes must already have Cobiveco or CobivecoX run on them
% (result.vtu with ab/rt/tm/tv point data, plus an R.mat alignment
% matrix). The source .vtu must additionally carry 'fiber' and 'sheet'
% point data (3-component vectors) -- see step 02c/04 for how to get
% those onto a Cobiveco result.
%
% Method (following Cobiveco's own examples/example_mapping_Cobiveco.m):
%   1. align both meshes to their own canonical frame (R.mat)
%   2. compute a UVC-based mapping matrix ('linear' = barycentric
%      interpolation in UVC space; use this whenever source is coarser
%      than target, which is the usual case for a template -> patient
%      mapping)
%   3. apply the mapping to the ALIGNED fiber/sheet vectors -- vector
%      data only makes sense to combine in a common frame
%   4. un-align the result back into the target's own coordinate frame
%   5. re-orthonormalize (barycentric combinations of unit vectors don't
%      stay unit length or exactly orthogonal)
%
% Inputs:
%   sourceVtuPath, targetVtuPath : Cobiveco/CobivecoX result.vtu paths
%   sourceRPath, targetRPath     : matching R.mat alignment matrix paths
%   outputPath                   : where to write the target mesh with
%                                   mapped fiber/sheet/normal + a
%                                   fiberValid flag (0 where no source
%                                   match was found, e.g. outside the
%                                   source mesh's UVC coverage)
%   cobivecoRoot                 : path to your Cobiveco installation
%                                   (default: '~/Documents/PhD/Cobiveco')
%   searchradius                 : optional, default 2 (see
%                                   cobiveco_computeMappingMatrix.m)
%
% Example:
%   map_fibers_uvc('template_with_fibers.vtu', 'result_cut/result.vtu', ...
%       'result_template/R.mat', 'result_cut/R.mat', 'cut_with_fibers.vtu')

if nargin < 6 || isempty(cobivecoRoot)
    cobivecoRoot = fullfile(getenv('HOME'), 'Documents', 'PhD', 'Cobiveco');
end
if nargin < 7 || isempty(searchradius)
    searchradius = 2;
end

addpath(fullfile(cobivecoRoot, 'functions'));
addpath(fullfile(cobivecoRoot, 'utilities'));
addpath(fullfile(cobivecoRoot, 'utilities', 'inputPreparation'));
addpath(fullfile(cobivecoRoot, 'dependencies', 'gptoolbox', 'matrix'));
addpath(fullfile(cobivecoRoot, 'dependencies', 'gptoolbox', 'mesh'));
addpath(fullfile(cobivecoRoot, 'dependencies', 'vtkToolbox', 'MATLAB'));

disp('Loading source and target...');
source = vtkRead(sourceVtuPath);
target = vtkRead(targetVtuPath);
fprintf('  source: %d points, %d cells\n', size(source.points,1), size(source.cells,1));
fprintf('  target: %d points, %d cells\n', size(target.points,1), size(target.cells,1));

if ~isfield(source.pointData, 'fiber') || ~isfield(source.pointData, 'sheet')
    error('Source mesh must have pointData.fiber and pointData.sheet (see step 04).');
end

R_source = load(sourceRPath); R_source = R_source.R;
R_target = load(targetRPath); R_target = R_target.R;

disp('Aligning both meshes to their own canonical frame...');
source_aligned = cobiveco_applyAlignmentMatrix(source, R_source);
target_aligned = cobiveco_applyAlignmentMatrix(target, R_target);

disp('Computing UVC-based mapping matrix (linear)...');
M = cobiveco_computeMappingMatrix(source_aligned, target_aligned, 'linear', searchradius, true);

nUnmapped = sum(isnan(full(sum(M,2))) | full(sum(M,2))==0);
fprintf('Target points with no valid source match: %d / %d\n', nUnmapped, size(target.points,1));

disp('Mapping fiber/sheet vectors (rotate to common frame, map, rotate back)...');
R3_source = R_source(1:3,1:3);
R3_target = R_target(1:3,1:3);

fiber_src_aligned = double(source.pointData.fiber) * R3_source';
sheet_src_aligned = double(source.pointData.sheet) * R3_source';

fiber_tgt_aligned = M * fiber_src_aligned;
sheet_tgt_aligned = M * sheet_src_aligned;

fiber_tgt = fiber_tgt_aligned * R3_target;
sheet_tgt = sheet_tgt_aligned * R3_target;

disp('Re-orthonormalizing...');
fiber_norm = sqrt(sum(fiber_tgt.^2,2));
fiber_tgt = fiber_tgt ./ fiber_norm;
sheet_tgt = sheet_tgt - sum(sheet_tgt.*fiber_tgt,2).*fiber_tgt;
sheet_norm = sqrt(sum(sheet_tgt.^2,2));
sheet_tgt = sheet_tgt ./ sheet_norm;
normal_tgt = cross(fiber_tgt, sheet_tgt, 2);

badRows = isnan(fiber_norm) | (fiber_norm < 1e-6);
fprintf('Rows with degenerate/NaN fiber after mapping: %d\n', sum(badRows));

target.pointData.fiber = fiber_tgt;
target.pointData.sheet = sheet_tgt;
target.pointData.normal = normal_tgt;
target.pointData.fiberValid = double(~badRows & ~isnan(full(sum(M,2))) & full(sum(M,2))~=0);

vtkWrite(target, outputPath);
fprintf('\nWrote %s: %d points\n', outputPath, size(target.points,1));
disp('Done.');

end
