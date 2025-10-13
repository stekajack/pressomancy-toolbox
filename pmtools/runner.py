import pmtools.refractored_toolbox as context
from pmtools.resources.kernel_config import AnalysisConfig
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
    """
    Decorator to wrap engine methods with fail-safe error handling.

    Any exception raised by the wrapped method is logged, the engine is
    shut down via ``self.shutdown()``, and the exception is re-raised.

    Parameters
    ----------
    method : callable
        Method to wrap.

    Returns
    -------
    callable
        Wrapped method that provides error logging and cleanup.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logging.error(f"safe_method_call caught exception in '{method.__name__}': {e}")
            self.shutdown()
            raise
    return wrapper

class Engine():
    """
    Lightweight parallel execution engine for analysis kernels.

    The engine coordinates assembling file paths, spawning processes to run
    registered kernel functions, tracking progress with a live progress bar,
    collecting results, and persisting them to disk.

    Attributes
    ----------
    global_max_workers : int
        Global hard limit for the sum of ``max_workers`` across active engines.
    max_workers_sum : int
        Class-wide running total of registered workers.
    event_horison : list
        Global list of unique future tags to prevent duplicate submissions.

    Notes
    -----
    Use as a context manager (``with Engine(world_path) as eng: ...``) to
    ensure resources are cleaned up automatically.
    """
    global_max_workers = 16
    max_workers_sum = 0
    event_horison = list()

    def __init__(self, world_path):
        """
        Initialize the engine with a working directory path.

        Parameters
        ----------
        world_path : str
            Root directory prepended to all assembled relative paths.
        """
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
        """Enter the runtime context and return ``self``."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and shut down resources."""
        self.shutdown()

    def register_kernel(self, function_handle, **kwargs):
        """
        Register a kernel function to be executed by the engine.

        Parameters
        ----------
        function_handle : callable
            A callable that accepts a single ``AnalysisConfig`` argument.
        **kwargs
            Keyword arguments to embed into the produced ``AnalysisConfig``
            when this kernel is invoked.

        Raises
        ------
        ValueError
            If ``function_handle`` is not callable or is already registered.
        """
        if not callable(function_handle):
            raise ValueError("Provided function handle is not callable.")
        if function_handle in self.functions_to_call:
            raise ValueError(f"A function named '{function_handle.__name__}' is already registered.")
        self.functions_to_call.append(function_handle)
        self.kernel_kwargs[function_handle.__name__] = kwargs

    def assemble_paths(self, master_dict, template_hndl, parallel_param_id=None):
        """
        Assemble path keys and execution paths from a template and parameter grid.

        Parameters
        ----------
        master_dict : dict
            Mapping from placeholder name to iterable of values.
        template_hndl : string.Template
            Template used to format path strings.
        parallel_param_id : str, optional
            Name of a parameter to expand in parallel per assembled base key.

        Notes
        -----
        Results are stored in ``self._keys_assembly`` and
        ``self._paths_accordingly`` for subsequent execution.
        """
        if self.template_hndl is None:
            self.template_hndl = template_hndl
        self._keys_assembly, self._paths_accordingly = context.assemble_paths(master_dict, template_hndl, parallel_param_id)

    @safe_method_call
    def run(self, max_workers=None):
        """
        Launch registered kernels over the assembled paths using a process pool.

        Parameters
        ----------
        max_workers : int, optional
            Number of worker processes for this engine instance. If omitted,
            the existing ``self.max_workers`` value is used.

        Raises
        ------
        RuntimeError
            If the cumulative workers across engines exceed ``global_max_workers``
            or if duplicate futures are attempted for the same path.
        """
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
                    cfg=AnalysisConfig(
                        data_path=loc_path,
                        template_hndl=self.template_hndl,**self.kernel_kwargs[function_handle.__name__])
                    future = self._executor_pool_hndl.submit(function_handle, cfg)
                    futures[string_id].append(future)
                    self._flat_future_list.append(future)
            self._pool_global[function_handle.__name__] = futures
            self.start_time = time.time()

    def track_progress_pretty(self):
        """
        Render a live textual progress bar and handle task exceptions.

        This method monitors ``self._flat_future_list``, periodically updates
        a progress bar with elapsed time and ETA, and stops when all tasks are
        complete. If any future raises an exception, it is logged, the engine
        is shut down, and the exception is re-raised.

        Raises
        ------
        RuntimeError
            If called before any futures are registered (i.e., before ``run``).
        """
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
    def collect_results(self):
        """
        Block until all tasks complete and collect their return values.

        Returns
        -------
        dict
            Nested mapping ``{kernel_name: {key: [result, ...], ...}, ...}``
            where values are ordered lists of results per assembled assignment.
        """
        logging.info('Collecting results...')
        self.track_progress_pretty()
        for key, elems in self._pool_global.items():
            for assignment, assignment_futures in elems.items():
                self._pool_global[key][assignment] = [future.result()
                                                      for future in assignment_futures]
        logging.info("Results collected.")
        return self._pool_global

    @safe_method_call
    def save_results(self, filename, custom_full_path=None):
        """
        Serialize and save collected results as a compressed pickle.

        Parameters
        ----------
        filename : str
            Base filename (without extension) used when ``custom_full_path``
            is not provided. The file is saved as ``<filename>.p.gz``.
        custom_full_path : str, optional
            Full path (including filename) to write the gzip-pickled object.

        Notes
        -----
        Data saved is the current ``self._pool_global``.
        """
        path = custom_full_path if custom_full_path is not None else os.path.join(self._work_dir, f'{filename}.p.gz')
        with gzip.open(path, 'wb') as f:
            pickle.dump(self._pool_global, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Results saved to {path}")

    def shutdown(self):
        """
        Cancel outstanding futures, stop the executor, and reset state.

        Notes
        -----
        If no executor is active, a warning is logged and no action is taken.
        """
        if self._executor_pool_hndl is None:
            logging.warning("Pool is empty.")
            return
        self._executor_pool_hndl.shutdown(wait=False, cancel_futures=True)
        self._executor_pool_hndl = None
        self.reset()
        logging.info("Runners in Pool closed.")

    @safe_method_call
    def reset(self):
        """
        Reset the engine to a clean state.

        De-registers kernels and paths, clears futures and global tags, and
        updates the class-wide worker accounting. Must not be called while an
        executor is active.

        Raises
        ------
        RuntimeError
            If called while the executor is still running.
        """
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