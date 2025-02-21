This code builds the MetaSRA. It uses Condor to parallelize mapping samples
to ontology terms. The pipeline can be run using Snakemake. Each step of the
pipeline is documented in the Snakefile. 

The Snakemake pipeline:

![dag](https://github.com/deweylab/MetaSRA-pipeline/blob/master/create_metasra/dag.png)

### Running the snakemake command

We recommend using some cluster-like execution environment, specifically, we use HTCondor.
We also use Conda for steps requiring Python 2.

Using HTCondor and Conda
```
snakemake --profile htcondor --use-conda
```

