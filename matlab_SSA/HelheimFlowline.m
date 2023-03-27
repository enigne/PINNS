clear
% inversion for Helheim flowline
ISSMpath = issmdir();
% load flowline data
disp('	Loading flowline data');
flowline = load('./DATA/Helheim_Weertman_iT080_PINN_flowline_CF.mat');
nx = 200;
ny = 50;
Lx = max(flowline.x);
Ly = 2e4;
yts = 3600*24*365;
% extrude x, y, u, H, h, etc in y direction
x = flowline.x;
y = linspace(0, Ly, ny)';
vx = repmat(flowline.vel', ny, 1).*yts;
vy = zeros(size(vx));
h = repmat(flowline.h', ny, 1);
H = repmat(flowline.H', ny, 1);
C = repmat(flowline.C', ny, 1);
bed = h - H;

% make it to a 2D plane {{{
% mesh
disp('	Creating 2D mesh');
md = squaremesh(model(), Lx, Ly, nx, ny);
% project data
disp('	Projecting flowline data to 2D mesh');
md.initialization.vx = InterpFromGrid(x,y,vx,md.mesh.x, md.mesh.y);
md.initialization.vy = InterpFromGrid(x,y,vy,md.mesh.x, md.mesh.y);
md.geometry.bed = InterpFromGrid(x,y,bed,md.mesh.x, md.mesh.y);
md.geometry.base = InterpFromGrid(x,y,bed,md.mesh.x, md.mesh.y);
md.geometry.surface = InterpFromGrid(x,y,h,md.mesh.x, md.mesh.y);

% set masks
disp('	Setting ice mask');
icemask = -1*ones(size(md.mesh.x));
icemask(md.mesh.x>flowline.cx) = 1;
md.mask.ice_levelset = icemask;
pos = find(md.mask.ice_levelset>0);
md.geometry.surface(pos) = md.geometry.base(pos)+10; %Minimum thickness
md.geometry.thickness = md.geometry.surface - md.geometry.bed;
pos=find(md.geometry.thickness<=10);
md.geometry.surface(pos) = md.geometry.base(pos)+10; %Minimum thickness
md.geometry.thickness = md.geometry.surface - md.geometry.bed;
md.masstransport.min_thickness = 10;
disp('	Adjusting ice mask');
%Tricky part here: we want to offset the mask by one element so that we don't end up with a cliff at the transition
pos = find(max(md.mask.ice_levelset(md.mesh.elements),[],2)>0);
md.mask.ice_levelset(md.mesh.elements(pos,:)) = 1;
% For the region where surface is NaN, set thickness to small value (consistency requires >0)
pos=find((md.mask.ice_levelset<0).*(md.geometry.surface<0));
md.mask.ice_levelset(pos)=1;
pos=find((md.mask.ice_levelset<0).*(isnan(md.geometry.surface)));
md.mask.ice_levelset(pos)=1;

disp('      -- reconstruct thickness');
md.geometry.thickness=md.geometry.surface-md.geometry.base;

disp('      reading velocities ');
md.inversion.vx_obs = md.initialization.vx;
md.inversion.vy_obs = md.initialization.vy;
md.inversion.vel_obs  = sqrt(md.inversion.vx_obs.^2+md.inversion.vy_obs.^2);
md.initialization.vz  = zeros(md.mesh.numberofvertices,1);
md.initialization.vel = md.inversion.vel_obs;

disp('   Initialize basal friction using driving stress');
disp('   -- Smooth the ice surface with 20 L2 projections and then compute the surface slopes');
asurf    = averaging(md,md.geometry.surface,20); % maybe executing 20 L2 projection is ok
[sx,sy,s]= slope(md,asurf); % slope 's' comes on elements
sslope   = averaging(md,s,1); % average the slope once on the vertices, because 's' comes on elements, we need this data on vertices

% set rheology
disp('   Creating flow law parameters (assume ice is at -5°C for now)');
md.materials.rheology_n = 3*ones(md.mesh.numberofelements,1);
md.materials.rheology_B = flowline.mu .*ones(md.mesh.numberofvertices,1);

%Deal with boundary conditions:
disp('   Set Boundary conditions');
md.stressbalance.spcvx=NaN*ones(md.mesh.numberofvertices,1);
md.stressbalance.spcvy=NaN*ones(md.mesh.numberofvertices,1);
md.stressbalance.spcvz=NaN*ones(md.mesh.numberofvertices,1);
md.stressbalance.referential=NaN*ones(md.mesh.numberofvertices,6);
md.stressbalance.loadingforce=0*ones(md.mesh.numberofvertices,3);
pos=find((md.mask.ice_levelset<0).*(md.mesh.vertexonboundary));
md.stressbalance.spcvx(pos)=md.initialization.vx(pos);
md.stressbalance.spcvy(pos)=md.initialization.vy(pos);
md.stressbalance.spcvz(pos)=0;


disp('   Initial basal friction ');
md.friction = frictionweertman();
md.friction.m = 3.0*ones(md.mesh.numberofelements,1);
ref_C = InterpFromGrid(x,y,C,md.mesh.x, md.mesh.y);
md.friction.C = 2000*ones(md.mesh.numberofvertices,1);

%No friction on PURELY ocean element
pos_e = find(min(md.mask.ice_levelset(md.mesh.elements),[],2)<0);
flags=ones(md.mesh.numberofvertices,1);
flags(md.mesh.elements(pos_e,:))=0;
md.friction.C(find(flags))=0.0;

md=setflowequation(md,'SSA','all');

md.mask.ocean_levelset = ones(md.mesh.numberofvertices,1);
md.cluster=generic('name',oshostname(),'np',40);
md.miscellaneous.name = ['test'];

%Control general
md.inversion=m1qn3inversion(md.inversion);
md.inversion.iscontrol=1;
md.verbose=verbose('solution',false,'control',true);
md.transient.amr_frequency = 0;

%Cost functions
md.inversion.cost_functions=[101 103 501];
md.inversion.cost_functions_coefficients=zeros(md.mesh.numberofvertices,numel(md.inversion.cost_functions));
md.inversion.cost_functions_coefficients(:,1)=1000;
md.inversion.cost_functions_coefficients(:,2)=180;
md.inversion.cost_functions_coefficients(:,3)=1.5e-8;
pos=find(md.mask.ice_levelset>0);
md.inversion.cost_functions_coefficients(pos,1:2)=0;

%Controls
md.inversion.control_parameters={'FrictionC'};
md.inversion.maxsteps=500;
md.inversion.maxiter =500;
md.inversion.min_parameters=0.01*ones(md.mesh.numberofvertices,1);
md.inversion.max_parameters=5e4*ones(md.mesh.numberofvertices,1);
md.inversion.control_scaling_factors=1;
md.inversion.dxmin = 1e-6;
%Additional parameters
md.stressbalance.restol=0.01;
md.stressbalance.reltol=0.1;
md.stressbalance.abstol=NaN;

md.toolkits.DefaultAnalysis=bcgslbjacobioptions();
md=solve(md,'Stressbalance');

ratio = [10,5,1];
% get a direction to update cost coeff
newCoeff = updateCostCoeff(md, ratio);
disp(sprintf('With the given ratio %d, %d, %d \n', ratio))
disp(sprintf('The coefficients can be updated to %g,   %g,   %g \n', newCoeff))

%Put results back into the model
md.friction.C=md.results.StressbalanceSolution.FrictionC;
md.initialization.vx=md.results.StressbalanceSolution.Vx;
md.initialization.vy=md.results.StressbalanceSolution.Vy;
%}}}
disp('  project friction.C to the central flowline')
x = flowline.x;
y = Ly/2*ones(size(x));
vel = flowline.vel;
C = InterpFromMeshToMesh2d(md.mesh.elements, md.mesh.x, md.mesh.y, md.friction.C, x, y);
h = flowline.h;
H = flowline.H;
mu = flowline.mu;
DBC = flowline.DBC;
icemask = flowline.icemask;
cx = flowline.cx;
nx = flowline.nx;
X_bc = flowline.X_bc;
u_bc = flowline.u_bc;
X_f = flowline.X_f;
icemask_f = flowline.icemask_f;

% plot
figure
subplot(3,1,1)
plot(x, vel)
subplot(3,1,2)
plot(x, h)
hold on
plot(x, h- H)
subplot(3,1,3)
plot(x, C)

% save
savefile = ['./DATA/Helheim_Weertman_iT080_PINN_flowline_CF_2dInv.mat'];
disp(['  Saving to ', savefile]);
save(savefile, ...
	'x', 'vel', 'C', 'h', 'H', 'mu', 'DBC', 'icemask', ...      % obs data
	'cx', 'nx', ...
	'X_bc', 'u_bc',...
	'X_f', 'icemask_f');

