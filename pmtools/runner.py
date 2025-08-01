import refractored_toolbox as context
import concurrent.futures
import os
import pickle
import gzip

class Engine():

    def __init__(self, world_path):
        self._work_dir = world_path
        self.max_workers = 10
        self.template_hndl=None
        self._keys_assembly=list()
        self._paths_accordingly=list()
        self.functions_to_call=list()
        self.kernel_kwargs=dict()
        self._pool_global = dict()
    
    def register_kernel(self, function_handle, **kwargs):
        if not callable(function_handle):
            raise ValueError("Provided function handle is not callable.")
        if function_handle in self.functions_to_call:
            raise ValueError(f"A function named '{function_handle.__name__}' is already registered.")
        self.functions_to_call.append(function_handle)
        self.kernel_kwargs[function_handle.__name__] = kwargs

    def assemble_paths(self, master_dict, template_hndl, parallel_param_id=None):
        if self.template_hndl is None:
            self.template_hndl = template_hndl
        self._keys_assembly,self._paths_accordingly=context.assemble_paths(master_dict, template_hndl, parallel_param_id)
    
    def run(self, max_workers=None):
        if max_workers is not None:
            self.max_workers = max_workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for function_handle in self.functions_to_call:
                futures={}
                for iid,string_id in enumerate(self._keys_assembly):
                    futures[string_id]=[executor.submit(function_handle, self._work_dir+path,self.template_hndl, **self.kernel_kwargs[function_handle.__name__]) for path in self._paths_accordingly[iid]]
                self._pool_global[function_handle.__name__] = futures
    
    def collect_results(self):
        for key, elems in self._pool_global.items():
            for assignment, assignment_futures in elems.items():
                self._pool_global[key][assignment] = [future.result()
                                                    for future in assignment_futures]
        return self._pool_global
    
    def save_results(self, filename, custom_full_path=None):
        path=None
        if custom_full_path is not None:
            path = custom_full_path
        else:
            path = os.path.join(self._work_dir, f'{filename}.p.gz')
        f = gzip.open(path, 'wb')
        pickle.dump(self._pool_global, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print(f"Results saved to {path}")
