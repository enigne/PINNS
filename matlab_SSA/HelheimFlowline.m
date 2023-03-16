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
x = repmat(flowline.x, ny, 1);
y = reshape(repmat(linspace(0, Ly, ny)', 1, length(flowline.x)), [], 1);
x = flowline.x;
y = linspace(0, Ly, ny)';
vx = repmat(flowline.vel', ny, 1).*yts;
vy = zeros(size(vx));
h = repmat(flowline.h', ny, 1);
H = repmat(flowline.H', ny, 1);
bed = h - H;

% make it to a 2D plane
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
disp('   -- Set the lower bound of velocity, pressure and friction coefficient');
min_velocity = 0;
min_pressure = 0;
min_friction_coef = 0.00;
disp('   -- Process surface velocity data');
vel      = md.inversion.vel_obs;
flags    = (vel==0).*(md.mask.ice_levelset<0); % interpolate on the ice parts
pos1     = find(flags);
pos2     = find(~flags);
vel(pos1)= griddata(md.mesh.x(pos2),md.mesh.y(pos2),vel(pos2),md.mesh.x(pos1),md.mesh.y(pos1)); % interpolating the velocities where vel==0
vel      = max(vel,min_velocity); % setting minimum velocity value
disp('   -- Calculate effective pressure and the initial pressure');
Neff                       = (md.materials.rho_ice*md.geometry.thickness+md.materials.rho_water*md.geometry.base)*md.constants.g;
Neff(find(Neff<=0))        = min_pressure; % setting minimum positve pressure
md.initialization.pressure = md.materials.rho_ice*md.geometry.thickness*md.constants.g; % setting the initial pressure
disp('   -- Deduce friction coefficient from driving stress');
driving_stress          = md.materials.rho_ice*md.constants.g*md.geometry.thickness.*(sslope);
md.friction.coefficient = sqrt(driving_stress./(Neff.*vel/md.constants.yts));
md.friction.coefficient = min(md.friction.coefficient,300);
md.friction.p           = ones(md.mesh.numberofelements,1);
md.friction.q           = ones(md.mesh.numberofelements,1);
md.friction.coupling = 2; % will be a default setting later, coupling=0 will give non-physical water pressure when above sea level.
disp('   -- Extrapolate on ice free regions (using griddata)');
flags = (md.mask.ice_levelset>0); % no ice
pos1  = find(flags);
pos2  = find(~flags);
md.friction.coefficient(pos1) = griddata(md.mesh.x(pos2),md.mesh.y(pos2),md.friction.coefficient(pos2),md.mesh.x(pos1),md.mesh.y(pos1));
pos = find(isnan(md.friction.coefficient) | md.friction.coefficient <=0);
md.friction.coefficient(pos)  = min_friction_coef;
% set the no ice area and negative effective pressure area to have minimal friction coef
md.friction.coefficient(pos1)  = min_friction_coef;
md.friction.coefficient(pos1)  = min_friction_coef;
md.friction.coefficient(Neff<=0)  = min_friction_coef;

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


return


md=parameterize(md,'./Par/SquareSheetShelf.par');
md=setflowequation(md,'SSA','all');
md=SetMarineIceSheetBC(md,'./Exp/SquareFront.exp');
L = 5e5;
md.friction.coefficient = ( 100+50*(sin(2*pi*md.mesh.x/L).*cos(2*pi*md.mesh.y/L)));
md.friction.p = 3*ones(md.mesh.numberofelements,1);
md.friction.q = zeros(md.mesh.numberofelements,1);
md.mask.ocean_levelset = ones(md.mesh.numberofvertices,1);

md.cluster=generic('name',oshostname(),'np',4);
md=solve(md,'Stressbalance');

x = md.mesh.x;
y = md.mesh.y;
H = md.geometry.thickness;
b = md.geometry.base;
vx = md.results.StressbalanceSolution.Vx ./ md.constants.yts;
vy = md.results.StressbalanceSolution.Vy ./ md.constants.yts;
C = md.friction.coefficient;

% compute the surface slopes
asurf = averaging(md,md.geometry.surface,20); % maybe executing 20 L2 projection is ok
[ssx, ssy] = computeGrad(md.mesh.elements, md.mesh.x, md.mesh.y, asurf); % compute the gradient
ssx = averaging(md,ssx,50); 
ssy = averaging(md,ssy,50); 
ss = sqrt(ssx.^2+ssy.^2);


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
	'ssx', 'ssy',...
	'cx', 'cy', 'nx', 'ny', 'smoothnx', 'smoothny', 'X_f', 'icemask_f');
plotmodel(md, 'data', md.results.StressbalanceSolution.Vel, 'data', C, 'data', vx, 'data', vy, 'data', md.mask.ice_levelset<0, 'data', md.mask.ocean_levelset<0)
