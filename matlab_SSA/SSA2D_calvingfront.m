%Test Name: SquareSheetShelfStressSSA2d
ISSMpath = issmdir();
md=triangle(model(),'./Exp/Square.exp',10000.);
md=parameterize(md,'./Par/SquareSheetShelf.par');
md=setflowequation(md,'SSA','all');
md=SetMarineIceSheetBC(md,'./Exp/SquareFront.exp');

md.cluster=generic('name',oshostname(),'np',4);
md=solve(md,'Stressbalance');

x = md.mesh.x;
y = md.mesh.y;
H = md.geometry.thickness;
b = md.geometry.base;
vx = md.results.StressbalanceSolution.Vx ./ md.constants.yts;
vy = md.results.StressbalanceSolution.Vy ./ md.constants.yts;
C = md.friction.coefficient;
C(md.mask.ocean_levelset<0) = 0.0;



% Dirichlet boundary
DBC = ~isnan(md.stressbalance.spcvx+md.stressbalance.spcvy);

% calving front boundary
contourline = isoline(md, md.mask.ice_levelset, 'value', 0);
tcx = contourline.x;
tcy = contourline.y;
% find the unique points only
utc = unique([tcx, tcy], 'rows');
tcx = utc(:,1); 
tcy = utc(:,2); 

cx = 0.5 * (tcx(1:end-1) + tcx(2:end));
cy = 0.5 * (tcy(1:end-1) + tcy(2:end));
nx = (tcy(1:end-1) - tcy(2:end));
ny = - (tcx(1:end-1) - tcx(2:end));
nn = sqrt(nx.^2+ny.^2);
nx = nx ./ nn;
ny = ny ./ nn;

% apply a filter
windowSize = 10;
wd = (1/windowSize)*ones(1,windowSize);
smoothnx = filter(wd,1,nx,[],1);
smoothny = filter(wd,1,ny,[],1);
% need to normalize again
nn = sqrt(smoothnx.^2+smoothny.^2);
smoothnx = smoothnx ./ nn;
smoothny = smoothny ./ nn;

plotmodel(md, 'data', 'BC')

hold on
for i = 1:length(cx)
	plot([cx(i), cx(i)+smoothnx(i)*1000],[cy(i), cy(i)+smoothny(i)*10000], 'r')
	plot([cx(i), cx(i)+nx(i)*1000],[cy(i), cy(i)+ny(i)*10000],'b--')
end
legend('off')

% set mask for only covered area
icemask = (md.mask.ice_levelset<0) & (~DBC);

% create collocation points
Xmin = min([x, y]);
Xmax = max([x, y]);
Nf = 1500;
X_ = Xmin + (Xmax - Xmin) .* lhsdesign(Nf, 2);
icemask_ = InterpFromMeshToMesh2d(md.mesh.elements,x,y, icemask+0, X_(:,1), X_(:,2));
X_f = X_(icemask_>0.5, :);
icemask_f = icemask_(icemask_>0.5);
% check the collocation points
plot(X_f(:,1), X_f(:,2), 'o')

save(['./DATA/SSA2D_calving.mat'], 'x', 'y', 'H', 'b', 'vx', 'vy', 'C', 'DBC', 'icemask',...
'cx', 'cy', 'nx', 'ny', 'smoothnx', 'smoothny', 'X_f', 'icemask_f');
plotmodel(md, 'data', md.results.StressbalanceSolution.Vel, 'data', C, 'data', vx, 'data', vy, 'data', md.mask.ice_levelset<0)
