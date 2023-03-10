clear
%Test Name: SquareSheetShelfStressSSA2d
ISSMpath = issmdir();
md=triangle(model(),'./Exp/HelheimSquare.exp',1000.);
md=parameterize(md,'./Par/HelheimSquare.par');
md=setflowequation(md,'SSA','all');
L = 8e4;

withCF = 1;
% change x and y to column vectors
x= md.mesh.x;
y= md.mesh.y;
% make cx and cy to row vectors
cy = L/2;
cx = L;
radius = L/8;
if withCF 
	savename = 'SSA2D_segCF';
else
	savename = 'SSA2D_seg_nocalving';
end

numberofvertices = md.mesh.numberofvertices;

% set levelset
ice_levelset = -ones(numberofvertices, 1);
if withCF
	%fjord = ((x-cx).^2+(y-cy).^2 < radius^2);
	%fjord = fjord | ((x>cx) & (y > cy-radius) & (y < cy+radius));
	fjord = (x>=cx) & (y > cy-radius) & (y < cy+radius);
	ice_levelset(fjord) = 0;
end

md.mask.ice_levelset = ice_levelset;
md.mask.ocean_levelset = ones(md.mesh.numberofvertices,1);
md = SetMarineIceSheetBC(md);

md.stressbalance.spcvx=NaN*ones(md.mesh.numberofvertices,1);
md.stressbalance.spcvy=NaN*ones(md.mesh.numberofvertices,1);
md.stressbalance.spcvz=NaN*ones(md.mesh.numberofvertices,1);
pos=find((md.mask.ice_levelset<0).*(md.mesh.vertexonboundary));
md.stressbalance.spcvx(pos)=md.initialization.vx(pos);
md.stressbalance.spcvy(pos)=md.initialization.vy(pos);
md.stressbalance.spcvz(pos)=0;

md.friction.coefficient = ( 300+50*(sin(4*pi*md.mesh.x/L).*cos(4*pi*md.mesh.y/L))) .* 10 .*(1-0.99*exp(-((y-L/2)/L*10).^2)) .* (exp(-5*(x/L).^2));
md.friction.p = 3*ones(md.mesh.numberofelements,1);
md.friction.q = zeros(md.mesh.numberofelements,1);

md.cluster=generic('name',oshostname(),'np',16);
md=solve(md,'Stressbalance');

%post processing{{{
H = md.geometry.thickness;
h = md.geometry.surface;
b = md.geometry.base;
vx = md.results.StressbalanceSolution.Vx ./ md.constants.yts;
vy = md.results.StressbalanceSolution.Vy ./ md.constants.yts;
C = md.friction.coefficient;

% Dirichlet boundary
DBC = ~isnan(md.stressbalance.spcvx+md.stressbalance.spcvy);
plotmodel(md, 'data', 'BC', 'figure', 1)

% calving front boundary
contourline = isoline(md, md.mask.ice_levelset, 'value', 0);
if ~isempty(contourline)
	tcx = contourline.x;
	tcy = contourline.y;
	% find the unique points only
	utc = unique([tcx, tcy], 'rows', 'stable');
	tcx = utc(:,1); 
	tcy = utc(:,2); 

	cx = 0.5 * (tcx(1:end-1) + tcx(2:end));
	cy = 0.5 * (tcy(1:end-1) + tcy(2:end));
	nx = - (tcy(1:end-1) - tcy(2:end));
	ny = (tcx(1:end-1) - tcx(2:end));
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

	hold on
	for i = 1:length(cx)
		plot([cx(i), cx(i)+smoothnx(i)*3000],[cy(i), cy(i)+smoothny(i)*3000], 'r')
		plot([cx(i), cx(i)+nx(i)*3000],[cy(i), cy(i)+ny(i)*3000],'b--')
	end
	legend('off')
else
	cx = 0;
	cy = 0;
	nx = 0;
	ny = 0;
	smoothnx = 0;
	smoothny = 0;
end

% set mask for only covered area
icemask = (md.mask.ice_levelset<0) & (~DBC);

% create collocation points
Xmin = min([x, y]);
Xmax = max([x, y]);
Nf = 10000;
X_ = Xmin + (Xmax - Xmin) .* lhsdesign(Nf, 2);
icemask_ = InterpFromMeshToMesh2d(md.mesh.elements,x,y, icemask+0, X_(:,1), X_(:,2));
X_f = X_(icemask_>0.5, :);
icemask_f = icemask_(icemask_>0.5);
% check the collocation points
hold on
plot(X_f(:,1), X_f(:,2), 'o')
hold off
%}}}
disp(['Saving to ', savename])
	save(['./DATA/', savename, '.mat'], 'x', 'y', 'h', 'H', 'vx', 'vy', 'C', 'DBC', 'icemask',...
		'cx', 'cy', 'nx', 'ny', 'smoothnx', 'smoothny', 'X_f', 'icemask_f');
plotmodel(md, 'data', md.results.StressbalanceSolution.Vel, 'data', C, 'data', vx, 'data', vy, 'data', md.mask.ice_levelset<0, 'data', md.mask.ocean_levelset<0, 'data', md.geometry.surface, 'data', md.geometry.thickness, 'figure', 2)
