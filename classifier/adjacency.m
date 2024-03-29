function A = adjacency(DATA, TYPE, PARAM, DISTANCEFUNCTION, SORT)

% Compute the adjacency graph of the data set DATA
%
% A = adjacency(DATA, TYPE, PARAM, DISTANCEFUNCTION);
%
% DATA - NxK matrix. Data points are rows.
% TYPE - string 'nn' or string 'epsballs'.
% PARAM - integer if TYPE='nn', real number if TYPE='epsballs'.
% DISTANCEFUNCTION - function mapping a (DxM) and a (D x N) matrix
%                    to an M x N distance matrix (D:dimensionality)
% SORT - Sort distances in ascend (DEFAULT) or descend
% Returns: A, sparse symmetric NxN matrix of distances between the
% adjacent points.
%
% Example:
%
% A = adjacency(X,'nn',6)
%   A contains the adjacency matrix for the data
%   set X. For each point, the distances to 6 adjacent points are
%   stored. N
%
% Note: the adjacency relation is symmetrized, i.e. if
% point a is adjacent to point b, then point b is also considered to be
% adjacent to point a.
%
%
% Author:
%
% Mikhail Belkin
% misha@math.uchicago.edu
%
% Modified by: Vikas Sindhwani
% June 2004

disp('Computing Adjacency Graph');

if (nargin < 3) || (strcmp(TYPE,'nn') && strcmp(TYPE,'epsballs')) || ~isreal(PARAM)
    fprintf('ERROR: Too few arguments given or incorrect arguments.\n\n');
    fprintf('USAGE:\n A = laplacian(DATA, TYPE, PARAM)\n');
    fprintf('DATA - the data matrix. Data points are rows.\n');
    fprintf('Nearest neigbors: TYPE =''nn''    PARAM = number of nearest neigbors\n');
    fprintf('Epsilon balls: TYPE =''epsballs''    PARAM = redius of the ball\n\n');
    return;
end
if (nargin < 5)
    SORT='ascend';
end
n = size(DATA,1);

A = sparse(n,n);
step = 100;

if (strcmp(TYPE,'nn'))
    for i1=1:step:n
        i2 = i1+step-1;
        if (i2> n)
            i2=n;
        end
        XX= DATA(i1:i2,:);
        dt = feval(DISTANCEFUNCTION, XX',DATA');
        [Z,I] = sort (dt,2,SORT);
        for i=i1:i2
            if ( mod(i, 500) ==0)
                %disp(sprintf('%d points processed.', i));
            end
            for j=2:PARAM+1
                A(i,I(i-i1+1,j))= Z(i-i1+1,j);
                A(I(i-i1+1,j),i)= Z(i-i1+1,j);
            end
        end
    end
    % epsilon balls
else
    for i1=1:step:n
        i2 = i1+step-1;
        if (i2> n)
            i2=n;
        end
        XX= DATA(i1:i2,:);
        dt = feval(DISTANCEFUNCTION, XX',DATA');
        [Z,I] = sort ( dt,2,SORT );
        for i=i1:i2
            %  if ( mod(i, 500) ==0) disp(sprintf('%d points processed.', i)); end;
            j=2;
            while ( (Z(i-i1+1,j) < PARAM))
                j = j+1;
                jj = I(i-i1+1,j);
                A(i,jj)= Z(i-i1+1,j);
                A(jj,i)= Z(i-i1+1,j);
            end
        end
    end
end