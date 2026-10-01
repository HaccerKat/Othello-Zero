import time
import torch.multiprocessing as mp

def worker(func, jobs, result_queue):
    for job in jobs:
        result = func(job)
        result_queue.put(result)

def execute_gpu(func, jobs):
    start = time.perf_counter()
    num_processes = min(len(jobs), 16)
    num_simulations = len(jobs)
    assert num_simulations % num_processes == 0, "Number of jobs must be divisible by number of processes"
    result_queue = mp.Queue()
    chunked_jobs = [jobs[i::num_processes] for i in range(num_processes)]

    processes = []
    for job_chunk in chunked_jobs:
        p = mp.Process(target=worker, args=(func, job_chunk, result_queue))
        p.start()
        processes.append(p)

    results = []
    for _ in range(num_simulations):
        results.append(result_queue.get())

    for p in processes:
        p.join()

    end = time.perf_counter()
    print(f"Executed {num_simulations} jobs in {end - start:.2f} seconds")
    return results