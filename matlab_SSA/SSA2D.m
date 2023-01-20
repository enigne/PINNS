%Test Name: SquareSheetConstrainedCMDragSSA2d
ISSMpath = issmdir();
md=triangle(model(),[ISSMpath, 'test/Exp/Square.exp'],20000.);
md=setmask(md,'','');
md=parameterize(md, './Par/SquareSheetConstrained.par');
md=setflowequation(md,'SSA','all');
% linear case first
L = 5e5;
md.friction.coefficient = 0* ( 100+50*(sin(2*pi*md.mesh.x/L).*cos(2*pi*md.mesh.y/L)));
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
DBC = md.mesh.vertexonboundary;

save(['./DATA/SSA2D_nofriction.mat'], 'x', 'y', 'H', 'b', 'vx', 'vy', 'C', 'DBC');
plotmodel(md, 'data', md.results.StressbalanceSolution.Vel, 'data', C, 'data', vx, 'data', vy)
