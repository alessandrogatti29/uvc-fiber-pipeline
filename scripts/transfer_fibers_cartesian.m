function transfer_fibers_cartesian(sourceVtuPath, targetVtuPath, outputPath, cobivecoRoot)
% Step 5: Transfer fiber/sheet/normal from a source mesh onto a target
% mesh via ordinary spatial (Cartesian) barycentric interpolation --
% NOT a UVC mapping. Use this when source and target are two different
% discretizations of the SAME real anatomy (e.g. a cut-down mesh and the
% original mesh it was cut from), so plain containing-tet interpolation
% is the right, simpler, more accurate tool -- no need to go through UVC
% when the geometries already agree in physical space.
%
% Target points that fall genuinely outside the source mesh's coverage
% (e.g. a region that was cut away from the source) are flagged rather
% than extrapolated to: they get fiberValid=0 and are left for a
% separate fill-in step (e.g. step 09's rule-based valve fibers).
%
% Inputs:
%   sourceVtuPath : mesh with pointData.fiber / pointData.sheet already set
%   targetVtuPath : mesh to transfer onto (same anatomy, different/larger mesh)
%   outputPath    : output .vtu
%   cobivecoRoot  : path to your Cobiveco installation, only needed for
%                   vtkRead/vtkWrite (default: '~/Documents/PhD/Cobiveco')
%
% Example:
%   transfer_fibers_cartesian('cut_with_fibers.vtu', 'full_mesh_with_valves.vtu', ...
%       'full_mesh_fibers_partial.vtu')

if nargin < 4 || isempty(cobivecoRoot)
    cobivecoRoot = fullfile(getenv('HOME'), 'Documents', 'PhD', 'Cobiveco');
end

addpath(fullfile(cobivecoRoot, 'functions'));
addpath(fullfile(cobivecoRoot, 'utilities'));
addpath(fullfile(cobivecoRoot, 'dependencies', 'vtkToolbox', 'MATLAB'));

disp('Loading source (with fiber/sheet) and target...');
source = vtkRead(sourceVtuPath);
target = vtkRead(targetVtuPath);
fprintf('  source: %d points, %d cells\n', size(source.points,1), size(source.cells,1));
fprintf('  target: %d points, %d cells\n', size(target.points,1), size(target.cells,1));

if ~isfield(source.pointData, 'fiber') || ~isfield(source.pointData, 'sheet')
    error('Source mesh must have pointData.fiber and pointData.sheet.');
end

P = double(source.points);
C = double(source.cells);
Q = double(target.points);

disp('Building triangulation and locating target points (this can take a while)...');
TR = triangulation(C, P);
baryCellIds = pointLocation(TR, Q);
found = ~isnan(baryCellIds);
fprintf('  target points found inside source mesh: %d / %d (%.1f%%)\n', ...
    sum(found), numel(found), 100*sum(found)/numel(found));

disp('Computing barycentric coordinates and interpolating fiber/sheet...');
baryCoords = nan(size(Q,1), 4);
baryCoords(found,:) = cartesianToBarycentric(TR, baryCellIds(found), Q(found,:));
baryPointIds = nan(size(Q,1), 4);
baryPointIds(found,:) = C(baryCellIds(found),:);

fiber_tgt = nan(size(Q,1), 3);
sheet_tgt = nan(size(Q,1), 3);
for k = 1:3
    vals = source.pointData.fiber(:,k);
    fiber_tgt(found,k) = sum(baryCoords(found,:) .* vals(baryPointIds(found,:)), 2);
    vals = source.pointData.sheet(:,k);
    sheet_tgt(found,k) = sum(baryCoords(found,:) .* vals(baryPointIds(found,:)), 2);
end

disp('Re-orthonormalizing...');
fiber_norm = sqrt(sum(fiber_tgt.^2,2));
fiber_tgt = fiber_tgt ./ fiber_norm;
sheet_tgt = sheet_tgt - sum(sheet_tgt.*fiber_tgt,2).*fiber_tgt;
sheet_norm = sqrt(sum(sheet_tgt.^2,2));
sheet_tgt = sheet_tgt ./ sheet_norm;
normal_tgt = cross(fiber_tgt, sheet_tgt, 2);

target.pointData.fiber = fiber_tgt;
target.pointData.sheet = sheet_tgt;
target.pointData.normal = normal_tgt;
target.pointData.fiberValid = double(found);

vtkWrite(target, outputPath);
fprintf('\nWrote %s: %d points, %d valid, %d need further fill-in\n', ...
    outputPath, size(Q,1), sum(found), sum(~found));
disp('Done.');

end
