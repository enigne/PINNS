%Test Name: SquareSheetConstrainedCMDragSSA2d
ISSMpath = issmdir();
md=triangle(model(),[ISSMpath, 'test/Exp/Square.exp'],200000.);
md=setmask(md,'','');
md=parameterize(md, './Par/SquareSheetConstrained.par');
md=setflowequation(md,'SSA','all');

%control parameters
md.inversion.iscontrol=1;
md.inversion.control_parameters={'FrictionCoefficient'};
md.inversion.min_parameters=1.*ones(md.mesh.numberofvertices,1);
md.inversion.max_parameters=200.*ones(md.mesh.numberofvertices,1);
md.inversion.nsteps=2;
md.inversion.cost_functions=[103  501];
md.inversion.cost_functions_coefficients=ones(md.mesh.numberofvertices,2); md.inversion.cost_functions_coefficients(:,2)=2.*10^-7;
md.inversion.gradient_scaling=3.*ones(md.inversion.nsteps,1);
md.inversion.maxiter_per_step=2*ones(md.inversion.nsteps,1);
md.inversion.step_threshold=0.3*ones(md.inversion.nsteps,1);
md.inversion.vx_obs=md.initialization.vx; md.inversion.vy_obs=md.initialization.vy;

md.cluster=generic('name',oshostname(),'np',3);
md=solve(md,'Stressbalance');

x = md.mesh.x;
y = md.mesh.y;
H = md.geometry.thickness;
b = md.geometry.bed;
vx = md.results.StressbalanceSolution.Vx;
vy = md.results.StressbalanceSolution.Vy;
C = md.results.StressbalanceSolution.FrictionCoefficient;

save(['./DATA/SSA2D.mat'], 'x', 'y', 'H', 'b', 'vx', 'vy', 'C');
