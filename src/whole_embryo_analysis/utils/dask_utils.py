import os
import pathlib
import datetime
import logging
import dask
import distributed
import dask_jobqueue
import psutil


logger = logging.getLogger(__name__)

# Default Dask-JobQueue cluster settings (can be overridden per-script)
DASK_DEFAULTS = {
    "workers": 1,
    "memory": "8G",
    "cores": 5,
    "processes": 1,
    "project": "beliveaulab",
    "queue": "beliveau-long.q",
    "runtime": "7200",
    "resource_spec": "mfree=8G",
    "conda_env": "otls-pipeline",
    "dashboard_port": ":41236",
    "log_dir": None,
}


def add_dask_args(parser, defaults=None):
    """
    Add --dask_* CLI arguments to an argparse parser.

    Args:
        parser: argparse.ArgumentParser instance
        defaults: dict overriding DASK_DEFAULTS for this script
    """
    d = {**DASK_DEFAULTS, **(defaults or {})}
    parser.add_argument("--dask_workers", type=int, default=d["workers"],
                        help="Number of Dask SGE workers")
    parser.add_argument("--dask_memory", type=str, default=d["memory"],
                        help="Memory per Dask worker (e.g. '8G')")
    parser.add_argument("--dask_cores", type=int, default=d["cores"],
                        help="CPU cores per Dask worker")
    parser.add_argument("--dask_processes", type=int, default=d["processes"],
                        help="Processes per Dask worker job")
    parser.add_argument("--dask_project", type=str, default=d["project"],
                        help="SGE project code")
    parser.add_argument("--dask_queue", type=str, default=d["queue"],
                        help="SGE queue name")
    parser.add_argument("--dask_runtime", type=str, default=d["runtime"],
                        help="SGE walltime (seconds or HH:MM:SS)")
    parser.add_argument("--dask_resource_spec", type=str, default=d["resource_spec"],
                        help="SGE resource specification (e.g. 'mfree=8G')")
    parser.add_argument("--dask_conda_env", type=str, default=d["conda_env"],
                        help="Conda environment for Dask workers")
    parser.add_argument("--dask_dashboard_port", type=str, default=d["dashboard_port"],
                        help="Dask dashboard port")
    parser.add_argument("--dask_log_dir", type=str, default=d["log_dir"],
                        help="Directory for Dask worker logs")


def dask_config_from_args(args):
    """
    Build the Dask-JobQueue config dict from parsed CLI args.

    Accepts either a Namespace or a dict (from vars(args)).
    Returns a dict with the keys expected by setup_dask_sge_cluster
    and SomiteAnalyzer.
    """
    if not isinstance(args, dict):
        args = vars(args)
    return {
        "num_workers": args["dask_workers"],
        "cpu_memory": args["dask_memory"],
        "cpu_cores": args["dask_cores"],
        "cpu_processes": args["dask_processes"],
        "cpu_resource_spec": args["dask_resource_spec"],
        "project": args["dask_project"],
        "queue": args["dask_queue"],
        "runtime": args["dask_runtime"],
        "log_dir": args.get("dask_log_dir"),
        "conda_env": args["dask_conda_env"],
        "dashboard_port": args["dask_dashboard_port"],
    }


def setup_dask_sge_cluster(
    n_workers: int,
    cores: int,
    processes: int,
    memory: str,
    project: str,
    queue: str,
    runtime: str, # Expecting format like 'HH:MM:SS' or seconds as string/int
    resource_spec: str | None = None,
    log_directory: str | None = None,
    conda_env: str | None = "dask-cellpose", # Default env name
    dashboard_port: str | None = ":12345",
    resources: bool | None = False,
    **kwargs # For extra SGECluster or dask config options
):
    """
    Sets up and configures a dask_jobqueue.SGECluster and a distributed.Client.

    Args:
        n_workers: Number of workers (SGE jobs) to request.
        cores: Number of cores per worker process.
        processes: Number of processes per worker job. Usually 1.
        memory: Memory per worker (e.g., '60GB').
        project: SGE project code.
        queue: SGE queue name.
        runtime: Job walltime (e.g., '08:00:00' or seconds).
        resource_spec: SGE resource specification (e.g., 'mfree=60G,gpgpu=1').
        log_directory: Directory for Dask worker logs.
        conda_env: Name of the conda environment to activate in the job script.
        kwargs: Additional keyword arguments passed to SGECluster or dask.config.set.

    Returns:
        A tuple containing the configured (dask_jobqueue.SGECluster, distributed.Client).
    """
    logger.info("Setting up Dask SGE Cluster...")

    # --- Dask Configuration ---
    dask_config_defaults = {
        'distributed.comm.timeouts.connect': '3600s',
        'distributed.comm.timeouts.tcp': '3600s',
        "distributed.worker.memory.spill": 0.70,
        "distributed.worker.memory.pause": 0.90,
        "distributed.worker.memory.terminate": 0.98,
        "distributed.scheduler.work-stealing": True,
        "distributed.scheduler.worker-saturation":  1.1,
        # Set nanny pre-spawn env variable if needed
        #"distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0
    }
    # Allow overriding defaults via kwargs
    dask_config = {**dask_config_defaults, **kwargs.pop('dask_config', {})}
    dask.config.set(dask_config)
    logger.info(f"Dask config applied: {dask_config}")

    # --- Log Directory ---
    if log_directory is None:
        log_directory = f"./dask_worker_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Dask worker logs will be stored in: {log_directory}")

    # --- Job Script Prologue ---
    prologue = []
    if conda_env:
        prologue.append(f'conda activate {conda_env}')
        prologue.append(f'export OMP_NUM_THREADS=1')
        prologue.append(f'export MKL_NUM_THREADS=1')
        prologue.append(f'export OPENBLAS_NUM_THREADS=1')
        prologue.append(f'export MALLOC_TRIM_THRESHOLD_=0')
    prologue.append(f'echo "Worker started on $(hostname) at $(date)"')

    # --- SGE PE Handling ---
    # dask-jobqueue usually handles cores via the PE if specified correctly
    # PE name might need to be configured depending on the cluster setup
    # Example: If PE 'smp' controls cores: job_extra_directives=['-pe smp ' + str(cores)]
    # If PE is not used for cores, dask handles it via --nthreads.
    sge_pe_name = 'serial' # Assuming 'serial' PE controls memory or other non-core resources
    job_extra = [f'-P {project} -pe serial {cores}']
    if resource_spec:
        # If resource_spec requests cores (e.g., 'gpgpu=1'), PE might be needed
        job_extra.append(f'-l {resource_spec}')
    # If cores > 1 and a specific PE is needed for multi-core jobs:
    # job_extra.append(f'-pe your_parallel_env {cores}')

    logger.info(f"Using project: {project}, queue: {queue}")
    logger.info(f"Requesting {cores} cores, {processes} processes, {memory} memory per worker.")
    logger.info(f"Job extra directives: {job_extra}")


    # --- Create Cluster ---
    cluster = dask_jobqueue.SGECluster(
        walltime=runtime,
        processes=processes,
        cores=cores,
        memory=memory,
        project=project,
        queue=queue,
        log_directory=log_directory,
        job_script_prologue=prologue,
        scheduler_options={"dashboard_address": dashboard_port},
        job_extra_directives=job_extra,
        worker_extra_args=["--resources", "GPU=1"] if resources else [],
        local_directory="$TMPDIR",
        # Add resource_spec if needed and not handled by job_extra
        # resource_spec=resource_spec if not any(rs in ' '.join(job_extra) for rs in ['mfree', 'gpgpu']) else None,
        # Pass remaining kwargs
        **kwargs
    )

    logger.info(f"Configured SGECluster. Requesting {n_workers} workers.")

    # --- Create Client and Scale ---
    client = distributed.Client(cluster)
    logger.info("Scaling cluster...")
    try:
        cluster.adapt(minimum_jobs=1, maximum_jobs=n_workers)
        logger.info(f"Cluster scale command issued. Waiting for workers (check dashboard).")

        # wait for workers to be ready 
        client.wait_for_workers(1, timeout=600)
        logger.info(f"Successfully connected to {len(client.scheduler_info()['workers'])} workers.")

        logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    except Exception as e:
        logger.error(f"Failed to scale cluster or connect client: {e}", exc_info=True)
        try:
            client.close()
            cluster.close()
        except Exception:
            pass # Ignore errors during cleanup after failure
        raise  # Re-raise the scaling exception

    return cluster, client

def adapt_cluster(cluster, min_workers, max_workers):
    _ = cluster.adapt(
        minimum_jobs=min_workers,
        maximum_jobs=max_workers,
        interval='10s',
        wait_count=6,
    )


def change_worker_attributes(
        cluster,
        min_workers,
        max_workers,
        **kwargs,
    ):
        """WARNING: this function is dangerous if you don't know what
           you're doing. Don't call this unless you know exactly what
           this does."""
        cluster.scale(0)
        for k, v in kwargs.items():
            cluster.new_spec['options'][k] = v
        adapt_cluster(cluster, min_workers, max_workers)

def shutdown_dask(cluster, client):
    """Gracefully shuts down the Dask client and cluster."""
    logger.info("Shutting down Dask client and cluster...")
    try:
        client.close()
        logger.info("Dask client closed.")
    except Exception as e:
        logger.error(f"Error closing Dask client: {e}", exc_info=True)
    try:
        cluster.close() # This terminates the SGE jobs
        logger.info("Dask cluster closed.")
    except Exception as e:
        logger.error(f"Error closing Dask cluster: {e}", exc_info=True)
    logger.info("Dask shutdown complete.") 

