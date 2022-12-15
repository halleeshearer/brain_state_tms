function pool = start_parpool(varargin)
   p = inputParser;
   p.addParameter('JobStorageLocation','',@ischar);
   p.addParameter('numCores',[],@isnumeric);
   p.parse(varargin{:});
   inputs = p.Results;

   cluster = parcluster;
   if isempty(inputs.numCores)
       cluster.NumWorkers = feature('numCores');
   else
       cluster.NumWorkers = inputs.numCores;
   end
   if ~isempty(inputs.JobStorageLocation)
      if ~isfolder(inputs.JobStorageLocation)
          mkdir(inputs.JobStorageLocation)
      end
      cluster.JobStorageLocation = inputs.JobStorageLocation;
   end
   pool = gcp('nocreate');
   if ~isempty(pool) && (pool.NumWorkers ~= cluster.NumWorkers || ~strcmp(cluster.JobStorageLocation,pool.Cluster.JobStorageLocation))
       delete(pool);
       pool = gcp('nocreate');
   end
   if isempty(pool)
       pool = parpool(cluster,cluster.NumWorkers);
   end
end 