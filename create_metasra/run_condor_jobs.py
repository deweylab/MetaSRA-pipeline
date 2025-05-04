from __future__ import print_function
from io import open # Python 2/3 compatibility
import os
import sys
import time

import query_condor_jobs
import condor_submit_tools

RECHECK = 60

def main():
    condor_root = sys.argv[1]
    submit_fname = sys.argv[2]
    finish_fname = sys.argv[3]
    run_condor_jobs(condor_root, submit_fname, finish_fname)

def run_condor_jobs(condor_root, submit_fname, finish_fname, cluster_id=None):
    print("Running Condor jobs...")
    cwd = os.getcwd()
    os.chdir(condor_root)
    if not cluster_id:
        num_jobs, cluster_id = condor_submit_tools.submit(submit_fname)
    jobs_still_going = True
    while jobs_still_going:
        print("Checking if any jobs are still running...")
        job_ids = query_condor_jobs.get_job_ids(cluster_id)
        print("Job ids returned by query: %s" % job_ids)
        if len(job_ids) == 0: # No more jobs in this cluster
            jobs_still_going = False
        else:
            jobs_still_going = False
            for job_id in job_ids:
                #job_id = "%s.%s" % (str(cluster_id), str(i))
                status = query_condor_jobs.get_job_status_in_queue(job_id)
                if status != "H":
                    print("Found job %s with status %s. Will keep checking..." % (job_id, status))
                    jobs_still_going = True
                    break
            time.sleep(RECHECK)
    print("No jobs were found running. Finished.")
    os.chdir(cwd)
    with open(finish_fname, 'w') as f:
        f.write('Finished.')

if __name__ == "__main__":
    main()
