%Test Name: SquareSheetConstrainedCMDragSSA2d
ISSMpath = issmdir();
md=triangle(model(),[ISSMpath, 'test/Exp/Square.exp'],20000.);
md=setmask(md,'','');
md=parameterize(md, './Par/SquareSheetConstrained.par');
md=setflowequation(md,'SSA','all');
% linear case first
L = 5e5;
md.friction.coefficient = ( 15+5*(sin(2*pi*md.mesh.x/L).*cos(2*pi*md.mesh.y/L)));
md.friction.p = 3*ones(md.mesh.numberofelements,1);
md.friction.q = zeros(md.mesh.numberofelements,1);

%control parameters
md.cluster=generic('name',oshostname(),'np',4);
md=solve(md,'Stressbalance');

x = md.mesh.x;
y = md.mesh.y;
H = md.geometry.thickness;
b = md.geometry.bed;
vx = md.results.StressbalanceSolution.Vx ./ md.constants.yts;
vy = md.results.StressbalanceSolution.Vy ./ md.constants.yts;
C = md.friction.coefficient;

% Dirichlet boundary
DBC = ~isnan(md.stressbalance.spcvx+md.stressbalance.spcvy);

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

cx = 0;
cy = 0;
nx = 0;
ny = 0;
smoothnx = 0;
smoothny = 0;

%save(['./DATA/SSA2D_nofriction.mat'], 'x', 'y', 'H', 'b', 'vx', 'vy', 'C', 'DBC');

save(['./DATA/SSA2D_friction.mat'], 'x', 'y', 'H', 'b', 'vx', 'vy', 'C', 'DBC', 'icemask',...
   'cx', 'cy', 'nx', 'ny', 'smoothnx', 'smoothny', 'X_f', 'icemask_f');

plotmodel(md, 'data', md.results.StressbalanceSolution.Vel, 'data', C, 'data', vx, 'data', vy)
