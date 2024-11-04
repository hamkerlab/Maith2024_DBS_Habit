from time import sleep
import subprocess
import concurrent.futures
import signal
from typing import List
import threading


def run_script_parallel(
    script_path: str, n_jobs: int, args_list: list = [""], n_total: int = 1
):
    """
    Run a script in parallel.

    Args:
        script_path (str):
            Path to the script to run.
        n_jobs (int):
            Number of parallel jobs.
        args_list (list, optional):
            List of lists containing the arguments (string values) of each run to pass
            to the script. Length of the list is the number of total runs. If a list
            of strings is passed these arguments are passed to the script and it is run
            n_total times. Default: [""], i.e. no arguments are passed to the script.
        n_total (int, optional):
            Number of total runs, only used if args_list is not a list of lists.
            Default: 1.
    """
    ### check if args_list is a list of lists
    if not isinstance(args_list[0], list):
        args_list = [args_list] * n_total
    elif n_total != 1:
        print(
            "run_script_parallel; Warning: n_total is ignored because args_list is a list of lists"
        )

    ### do not use more jobs than necessary
    n_jobs = min(n_jobs, len(args_list))

    ### run the script in parallel
    runner = _ScriptRunner(
        script_path=script_path, num_workers=n_jobs, args_list=args_list
    )
    runner.run()


class _ScriptRunner:
    def __init__(self, script_path: str, num_workers: int, args_list: List[List[str]]):
        self.script_path = script_path
        self.args_list = args_list
        self.num_workers = num_workers
        self.processes = []
        self.executor = None
        self.error_flag = threading.Event()

    def run_script(self, args: List[str]):
        """
        Run the script with the given arguments.

        Args:
            args (List[str]):
                List of arguments to pass to the script.
        """
        process = subprocess.Popen(
            ["python", self.script_path] + args,
        )
        self.processes.append(process)

        process.wait()
        # Check if the process returned an error
        if process.returncode != 0:
            self.error_flag.set()
            return -1

    def signal_handler(self, sig, frame):
        """
        Signal handler to terminate all running processes and shutdown the executor.
        """
        # need a small sleep here, otherwise a single new process is started, don't know why
        sleep(0.01)
        # Terminate all running processes
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        # Shutdown the executor
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
        # Exit the program
        exit(1)

    def run(self):
        """
        Run the script with the given arguments in parallel.
        """
        # Register the signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)
        # Create a thread pool executor with the specified number of workers
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        )
        # Submit the tasks to the executor
        try:
            futures = [
                self.executor.submit(self.run_script, args) for args in self.args_list
            ]
            # Wait for all futures to complete
            while any(f.running() for f in futures):
                # Check if an error occurred in any of the threads
                if self.error_flag.is_set():
                    self.signal_handler(None, None)
                    break
        finally:
            self.executor.shutdown(wait=True)
