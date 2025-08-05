import pmtools.refractored_toolbox as context
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import gzip
import sys
import time
import threading

class Engine():
    """
    Engine class for managing parallel execution of registered kernel functions
    over assembled paths, tracking progress, collecting and saving results.
    """

    global_max_workers = 16
    max_workers_sum = 0
    event_horison=list()

    def __init__(self, world_path):
        """
        Initialize the Engine instance.

        Args:
            world_path (str): Path to the working directory.
        """
        self._work_dir = world_path
        self.max_workers = 10
        self.template_hndl=None
        self._keys_assembly=list()
        self._paths_accordingly=list()
        self._flat_future_list=list()
        self.functions_to_call=list()
        self.kernel_kwargs=dict()
        self._pool_global = dict()
        self._executor_pool_hndl = None
        self.start_time=0.

    def register_kernel(self, function_handle, **kwargs):
        """
        Register a kernel function to be executed in parallel.

        Args:
            function_handle (callable): The function to register.
            **kwargs: Keyword arguments to pass to the function.

        Raises:
            ValueError: If the function is not callable or already registered.
        """
        if not callable(function_handle):
            raise ValueError("Provided function handle is not callable.")
        if function_handle in self.functions_to_call:
            raise ValueError(f"A function named '{function_handle.__name__}' is already registered.")
        self.functions_to_call.append(function_handle)
        self.kernel_kwargs[function_handle.__name__] = kwargs

    def assemble_paths(self, master_dict, template_hndl, parallel_param_id=None):
        """
        Assemble keys and paths for parallel execution.

        Args:
            master_dict (dict): Master dictionary for path assembly.
            template_hndl: Template handle for path assembly.
            parallel_param_id: Optional parameter for parallelization.
        """
        if self.template_hndl is None:
            self.template_hndl = template_hndl
        self._keys_assembly,self._paths_accordingly=context.assemble_paths(master_dict, template_hndl, parallel_param_id)
    
    def run(self, max_workers=None):
        """
        Submit registered kernel functions for parallel execution.

        Args:
            max_workers (int, optional): Maximum number of workers to use.

        Raises:
            RuntimeError: If total max workers exceeds global limit or duplicate futures are created.
        """
        if max_workers is not None:
            self.max_workers = max_workers
        Engine.max_workers_sum += self.max_workers
        if Engine.max_workers_sum > Engine.global_max_workers:
            raise RuntimeError(f"Total max workers {Engine.max_workers_sum} exceeds global limit {Engine.global_max_workers}.")
        self._executor_pool_hndl = ProcessPoolExecutor(max_workers=self.max_workers)
        for function_handle in self.functions_to_call:
            futures={}
            for iid,string_id in enumerate(self._keys_assembly):
                futures[string_id]=[]
                for path in self._paths_accordingly[iid]:
                    loc_path=self._work_dir+path
                    future_def=f'{function_handle.__name__}::{loc_path}'
                    if future_def in Engine.event_horison:
                        raise RuntimeError(f"Multiple futures were attemped to be created for the same location {loc_path}.")
                    Engine.event_horison.append(future_def)
                    future=self._executor_pool_hndl.submit(function_handle, loc_path, self.template_hndl, **self.kernel_kwargs[function_handle.__name__])
                    futures[string_id].append(future)
                    self._flat_future_list.append(future)
            self._pool_global[function_handle.__name__] = futures
            self.start_time = time.time()
            
    def track_progress_pretty(self):
        """
        Track and display progress of running futures with a progress bar.

        Raises:
            RuntimeError: If no futures are registered.
        """
        if not hasattr(self, '_flat_future_list'):
            raise RuntimeError("No futures registered. Call run() before tracking progress.")
            
        def format_time(t):
            mins, secs = divmod(int(t), 60)
            return f"{mins:02}:{secs:02}"
        
        def render_loop():
            while True:
                with completed_lock:
                    current_completed = completed
                    elapsed = time.time() - self.start_time
                    if current_completed > 0:
                        time_per_task = elapsed / current_completed
                        eta = time_per_task * (total - current_completed)
                    else:
                        eta = 0
                    percentage = int(100 * current_completed / total)

                bar = "#" * (percentage // 2)
                line = (
                    f"\r[{bar:<50}] {percentage}% "
                    f"| Elapsed: {format_time(elapsed)} "
                    f"| ETA: {format_time(eta)}"
                )
                sys.stdout.write(line)
                sys.stdout.flush()

                if current_completed >= total:
                    break
                time.sleep(0.5)
        
        total = len(self._flat_future_list)
        completed = 0
        any_running = any(x.running() for x in self._flat_future_list)
        if any_running:
            completed_lock = threading.Lock()
            render_thread = threading.Thread(target=render_loop)
            render_thread.start()
            for _ in as_completed(self._flat_future_list):
                with completed_lock:
                    completed += 1
            render_thread.join()
        print("\n✅ All tasks completed.")

    def collect_results(self, kill_workers=True):
        """
        Collect results from all futures and optionally shut down workers.

        Args:
            kill_workers (bool): Whether to shut down the executor after collecting results.

        Returns:
            dict: Collected results from all kernel functions.
        """
        self.track_progress_pretty()
        for key, elems in self._pool_global.items():
            for assignment, assignment_futures in elems.items():
                self._pool_global[key][assignment] = [future.result()
                                                    for future in assignment_futures]
        if kill_workers:
            self.shutdown()
        print("Results collected.")
        return self._pool_global
    
    def save_results(self, filename, custom_full_path=None):
        """
        Save collected results to a compressed pickle file.

        Args:
            filename (str): Name of the file to save results.
            custom_full_path (str, optional): Custom full path for saving the file.
        """
        path=None
        if custom_full_path is not None:
            path = custom_full_path
        else:
            path = os.path.join(self._work_dir, f'{filename}.p.gz')
        f = gzip.open(path, 'wb')
        pickle.dump(self._pool_global, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print(f"Results saved to {path}")

    def shutdown(self):
        """
        Shut down the executor pool and release resources.
        """
        self._executor_pool_hndl.shutdown(wait=True)
        print("Runners in Pool closed.")
