function[r,varargout] = icc21(X,varargin)
%ICC21 Intraclass correlation coefficient afeter model ICC(2,1).
%   R = ICC21(X) returns the intraclass correlation coefficent computed
%   from the data matrix X. X is a n-by-w matrix with colums corresponding
%   to persons and rows corresponding to measues, respectively. Element Xji
%   is the response to the i-th measure given by the j-th person. ICC21
%   returns the intraclass correlation coefficient computed after the model
%   ICC(2,1) given in [1].

%   Reference:
%   [1] Shrout, P.E.; Fleiss, J.L. (1979). Intraclass correlations: Uses in
%   assessing rater reliability. Psych. Bulletin, 86(2), pp. 420-428
%   [2] McGraw, K.O.; Wong, S.P. (1996). Forming Inferences about some
%   Intraclass Correlation Coefficients. Psych. Methods, 1(1), pp. 30-46
%20070928, Thomas Zoeller, tzo@gmx.de

%Default values
alpha=0.05; %Significance level
testval=0; %Null hypothesis test value, H0: r=testval

%Evaluate input arguments
if nargin<1
    r=[];
    disp(['Error: not enough input arguments.']);
    return;
elseif nargin>1
    i=1;
    while i<length(varargin)
        switch lower(varargin{i})
            case 'alpha'
                alpha=varargin{i+1};
            case 'testval'
                testval=varargin{i+1};
        end
        i=i+1;
    end
end

%Compute ICC(2,1)
[p,table]=anova2(X,1,'off'); %Two-way random effects ANOVA model

MS_subjects=table{3,4}; %Measurements, corresponding to rows
MS_judges=table{2,4}; %Mean squares between judges, corresp. to the columns
MS_error=table{4,4}; %Residual mean squares
df_judges=table{2,3};
n_subjects=size(X,1);
n_judges=size(X,2);

r_num=(MS_subjects-MS_error);
r_den=(MS_subjects+df_judges*MS_error+n_judges*(MS_judges-MS_error)/n_subjects);
r=r_num/r_den;

%Compute confidence interval
k=n_judges;
W=n_subjects;

MSbp=table{3,4}; %Mean square rows
MSbm=table{2,4}; %Mean square columns
MSres=table{4,4}; %Mean square residual

a= (k*r) / (W*(1-r));
b= 1+( (k*r*(W-1)) / (W*(1-r)) );
v= ((a*MSbm+b*MSres)^2) / ( ( ((a*MSbm)^2) / (k-1) ) + ( ((b*MSres)^2) / ((W-1)*(k-1)) ) );

%Method after McGraw & Wong, given in [2]
n=W; %here: n=W
Flower=finv(alpha/2,n-1,v);
Fupper=finv(alpha/2,v,n-1);

%lower=(n*(bms-fsupstar*ems))/(fsupstar*(k*jms+(k*n-k-n)*ems)+n*bms)
rlower = (n*(MSbp-Flower*MSres)) / (Flower*(k*MSbm +(k*n-k-n)*MSres)+n*MSbp);
%upper=(n*(fsubstar*bms-ems))/ (k*jms+(k*n-k-n)*ems+n*fsubstar*bms)
rupper = (n*(Fupper*MSbp-MSres)) / (k*MSbm + MSres*(k*n-k-n)+n*Fupper*MSbp);

OUT{1}=[rupper rlower];

%Test of significance for H0
r0=testval;
a0= (k*r0) / (W*(1-r0));
b0= 1+( (k*r0*(W-1)) / (W*(1-r0)) );
v0= ((a0*MSbm+b0*MSres)^2) / ( ( ((a0*MSbm)^2) / (k-1) ) + ( ((b0*MSres)^2) / ((W-1)*(k-1)) ) );

Fr=MSbp/(a0*MSbm + b0*MSres);
p1=fcdf(Fr,W-1,v0);
p2=1-p1;

if p1<=p2
    if p1<alpha/2
        H=0; %Reject Null hypothesis
    else
        H=1; %Accept Null hypothesis
    end
    p=p1;
else
    if p2<alpha/2
        H=0; %Reject Null hypothesis
    else
        H=1; %Accept Null hypothesis
    end
    p=p2;
end

OUT{2}=H;
OUT{3}=p;

%Fill output arguments
if nargout>1
    for i=1:nargout-1
        varargout{i}=OUT{i};
    end
end
