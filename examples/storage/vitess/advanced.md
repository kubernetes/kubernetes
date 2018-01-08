# Advanced Vitess on Kubernetes

## Automatically run Vitess on Container Engine

The following commands will create a Google Container Engine cluster and bring
up Vitess with two shards and three tablets per shard: (Note that it does not
bring up the Guestbook example)

```

vitess/examples/kubernetes$ export SHARDS=-80,80-
vitess/examples/kubernetes$ export GKE_ZONE=us-central1-b
vitess/examples/kubernetes$ export GKE_NUM_NODES=10
vitess/examples/kubernetes$ export GKE_MACHINE_TYPE=n1-standard-8
vitess/examples/kubernetes$ ./cluster-up.sh
vitess/examples/kubernetes$ ./vitess-up.sh
```

Run the following to tear down the entire Vitess + container engine cluster:

```
vitess/examples/kubernetes$ ./vitess-down.sh
vitess/examples/kubernetes$ ./cluster-down.sh
```

## Parameterizing configs

The vitess and cluster scripts both support parameterization via exporting
environment variables.

### Parameterizing cluster scripts

Common environment variables:

* GKE_ZONE - Zone to use for Container Engine (default us-central1-b)
* GKE_CLUSTER_NAME - Name to use when creating a cluster (default example).
* SHARDS - Comma delimited keyranges for shards (default -80,80- for 2 shards).
Use 0 for an unsharded keyspace.

The cluster-up.sh script supports the following environment variables:

* GKE_MACHINE_TYPE - Container Engine machine type (default n1-standard-1)
* GKE_NUM_NODES - Number of nodes to use for the cluster (required).
* GKE_SSD_SIZE_GB - SSD size (in GB) to use (default 0 for no SSD).

The vitess-up.sh script supports the following environment variables:

* TABLETS_PER_SHARD - Number of tablets per shard (default 3).
* RDONLY_COUNT - Number of tablets per shard that are rdonly (default 0).
* VTGATE_COUNT - Number of vtgates (default 25% of total vttablet count,
with a minimum of 3).

For example, to create an unsharded keyspace with 5 tablets, use the following:

```
export SHARDS=0
export TABLETS_PER_SHARD=5
vitess/examples/kubernetes$ ./cluster-up.sh
vitess/examples/kubernetes$ ./vitess-up.sh
```

