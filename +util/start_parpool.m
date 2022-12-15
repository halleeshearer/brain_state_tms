function pool = start_parpool(numCores)
if ~exist('numCores','var') || ~isnumeric(numCores)
    numCores = feature('numCores');
end
pool = gcp('nocreate');
if ~isempty(pool) && pool.NumWorkers ~= numCores
    delete(pool);
    pool = gcp('nocreate');
end
if isempty(pool)
    pool = parpool(numCores);
end
end