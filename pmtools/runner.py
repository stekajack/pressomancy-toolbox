import pmtools.refractored_toolbox as context
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import gzip
import sys
import time
import threading
import numpy as np
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def safe_method_call(method):
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logging.error(f"safe_method_call caught exception in '{method.__name__}': {e}")
            self.shutdown()
            raise
    return wrapper

class Engine():
    global_max_workers = 16
    max_workers_sum = 0
    event_horison = list()

    def __init__(self, world_path):
        self._work_dir = world_path
        self.max_workers = 10
        self.template_hndl = None
        self._keys_assembly = list()
        self._paths_accordingly = list()
        self._flat_future_list = list()
        self.functions_to_call = list()
        self.kernel_kwargs = dict()
        self._pool_global = dict()
        self._executor_pool_hndl = None
        self.start_time = 0.
        self.local_event_horison_tags = list()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

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
        self._keys_assembly, self._paths_accordingly = context.assemble_paths(master_dict, template_hndl, parallel_param_id)

    @safe_method_call
    def run(self, max_workers=None):
        if max_workers is not None:
            self.max_workers = max_workers
        Engine.max_workers_sum += self.max_workers
        if Engine.max_workers_sum > Engine.global_max_workers:
            raise RuntimeError(f"Total max workers {Engine.max_workers_sum} exceeds global limit {Engine.global_max_workers}.")
        self._executor_pool_hndl = ProcessPoolExecutor(max_workers=self.max_workers)
        for function_handle in self.functions_to_call:
            futures = {}
            for iid, string_id in enumerate(np.atleast_1d(self._keys_assembly)):
                futures[string_id] = []
                for path in np.atleast_1d(self._paths_accordingly[iid]):
                    loc_path = self._work_dir + path
                    future_def = f'{function_handle.__name__}::{loc_path}'
                    if future_def in Engine.event_horison:
                        raise RuntimeError(f"Multiple futures were attempted to be created for the same location {loc_path}.")
                    Engine.event_horison.append(future_def)
                    self.local_event_horison_tags.append(future_def)
                    future = self._executor_pool_hndl.submit(function_handle, loc_path, self.template_hndl, **self.kernel_kwargs[function_handle.__name__])
                    futures[string_id].append(future)
                    self._flat_future_list.append(future)
            self._pool_global[function_handle.__name__] = futures
            self.start_time = time.time()

    def track_progress_pretty(self):
        if not hasattr(self, '_flat_future_list'):
            raise RuntimeError("No futures registered. Call run() before tracking progress.")

        def format_time(t):
            mins, secs = divmod(int(t), 60)
            return f"{mins:02}:{secs:02}"

        def render_loop():
            while not stop_flag.is_set():
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

        any_running = any(x.running() for x in self._flat_future_list)
        if any_running:
            total = len(self._flat_future_list)
            completed = sum(f.done() for f in self._flat_future_list)
            completed_lock = threading.Lock()
            render_thread = threading.Thread(target=render_loop)
            stop_flag = threading.Event()
            render_thread.start()
            for future in as_completed(self._flat_future_list):
                exc = future.exception()
                if exc is not None:
                    logging.error(f"Task raised an exception: {exc}")
                    stop_flag.set()
                    render_thread.join()
                    self.shutdown()
                    raise exc
                with completed_lock:
                    completed += 1
            render_thread.join()
            logging.info("✅ All tasks completed successfully.")

    @safe_method_call
    def collect_results(self, kill_workers=True):
        logging.info('Collecting results...')
        self.track_progress_pretty()
        for key, elems in self._pool_global.items():
            for assignment, assignment_futures in elems.items():
                self._pool_global[key][assignment] = [future.result()
                                                      for future in assignment_futures]
        if kill_workers:
            self.shutdown()
        logging.info("Results collected.")
        return self._pool_global

    @safe_method_call
    def save_results(self, filename, custom_full_path=None):
        path = custom_full_path if custom_full_path is not None else os.path.join(self._work_dir, f'{filename}.p.gz')
        with gzip.open(path, 'wb') as f:
            pickle.dump(self._pool_global, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Results saved to {path}")

    def shutdown(self):
        if self._executor_pool_hndl is None:
            logging.warning("Pool is empty.")
            return
        self._executor_pool_hndl.shutdown(wait=False, cancel_futures=True)
        self._executor_pool_hndl = None
        self.reset()
        logging.info("Runners in Pool closed.")

    @safe_method_call
    def reset(self):
        if self._executor_pool_hndl is not None:
            raise RuntimeError("Cannot reset while executor is running. Call shutdown() first.")
        Engine.max_workers_sum -= self.max_workers
        self.max_workers = 0
        self._pool_global.clear()
        self._flat_future_list.clear()
        for tag in np.atleast_1d(self.local_event_horison_tags):
            try:
                Engine.event_horison.remove(tag)
            except ValueError:
                pass
        self.local_event_horison_tags.clear()
        self.functions_to_call.clear()
        self.kernel_kwargs.clear()
        self._keys_assembly.clear()
        self._paths_accordingly.clear()
        self.template_hndl = None
        self.start_time = 0.
