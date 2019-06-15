from optparse import OptionParser
import os
from os.path import join, basename, realpath
import sys
import subprocess
import json
import math

import condor_submit_tools

BLACKLIST = [
    # TODO add machines you want to blacklist here
    # TODO this should eventually be placed in a config file
]

def main():
    parser = OptionParser()
    parser.add_option("-p", "--pipeline_index", help="The index of the pipeline to run")
    parser.add_option("-o", "--ontology_index", help="Index of the ontology to map to")
    parser.add_option("-r", "--pipeline_root", help="The pipeline's root")
    (options, args) = parser.parse_args()

    pipeline_root = args[0]
    submit_fname = args[1]
    condor_bundle_f = args[2]
    condor_exec_f = args[3]
    output_filename = args[4]
    sample_to_metadata_f = args[5]

    with open(sample_to_metadata_f, 'r') as f:
        sample_to_metadata = json.load(f)

    condorize_pipeline(
        pipeline_root, 
        submit_fname,
        condor_bundle_f,
        condor_exec_f,
        output_filename,
        sample_to_metadata
    )

    
def condorize_pipeline(
        pipeline_root, 
        submit_fname,
        condor_bundle_f, 
        condor_exec_f,
        output_filename,
        sample_to_tag_to_value
    ):
    # Create directory in which all job directories are located
    subprocess.call("mkdir %s" % pipeline_root, shell=True, env=None)

    submit_builder = condor_submit_tools.SubmitFileBuilder(
        condor_exec_f, 
        15000, 
        500, 
        op_sys_version="7",
        blacklist=BLACKLIST
    )

    # Create symlink to Condor executable
    subprocess.call(
        "ln -s %s %s" % (
            condor_exec_f, 
            join(pipeline_root, basename(condor_exec_f))
        ), 
        shell=True, 
        env=None
    )

    all_samples = sorted(sample_to_tag_to_value.keys())
    samp_chunks = _chunks(all_samples, int(math.ceil(len(all_samples) / 10000.0)))
    for i, s_chunk in enumerate(samp_chunks):

        # Create job directory
        job_dir = join(pipeline_root, "%d.root" % i)
        subprocess.call("mkdir %s" % job_dir, shell=True, env=None)

        # Create Condor input file
        j = {"sample_accessions": s_chunk}
        fname = "chunk_%d.json" % i
        list_f = join(job_dir, fname)
        with open(list_f, 'w') as f:
            f.write(json.dumps(j))

        # Copy input files into root directory
        in_files_locs = [
            condor_bundle_f,
            list_f, 
            condor_exec_f
        ]
        symlinks = []
        for in_f in in_files_locs:
            symlink = join(job_dir, basename(in_f))
            print "ln -s %s %s" % (in_f, symlink)
            subprocess.call("ln -s %s %s" % (in_f, symlink), shell=True, env=None)
            symlinks.append(symlink)

        submit_builder.add_job(
            job_dir, 
            arguments=[
                output_filename,
                "./%s" % basename(list_f) 
            ], 
            input_files=in_files_locs, 
            output_files=[output_filename]
        )

    content = submit_builder.build_file()
    with open(join(pipeline_root, submit_fname), "w") as f:
        f.write(content)

def _chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in xrange(0, len(l), n)]

if __name__ == "__main__":
    main()

