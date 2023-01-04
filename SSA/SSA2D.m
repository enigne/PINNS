%Test Name: SquareSheetConstrainedCMDragSSA2d
ISSMpath = issmdir();
md=triangle(model(),[ISSMpath, 'test/Exp/Square.exp'],20000.);
md=setmask(md,'','');
md=parameterize(md, './Par/SquareSheetConstrained.par');
md=setflowequation(md,'SSA','all');

%control parameters
md.cluster=generic('name',oshostname(),'np',4);
md=solve(md,'Stressbalance');

x = md.mesh.x;
y = md.mesh.y;
H = md.geometry.thickness;
b = md.geometry.bed;
vx = md.results.StressbalanceSolution.Vx;
vy = md.results.StressbalanceSolution.Vy;
C = md.friction.coefficient;

save(['./DATA/SSA2D.mat'], 'x', 'y', 'H', 'b', 'vx', 'vy', 'C');
