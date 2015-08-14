## High Availability

### Scheduler

The implementation of the scheduler HA feature includes:

- Checkpointing by default (`--checkpoint`)
- Large failover-timeout by default (`--failover-timeout`)
- Hot-failover w/ multiple scheduler instances (`--ha`)
- Best effort task reconciliation on failover

#### Multiple Instances

Multiple scheduler instances may be run to support a warm-standby scenario in which one scheduler fails and another takes over immediately.
But at any moment in time only one scheduler is actually registered with the leading Mesos master.
Scheduler leader election is implemented using etcd so it is important to have an HA etcd configuration established for reliable scheduler HA.

It is currently recommended that no more than 2 scheduler instances be running at the same time.
Running more than 2 schedulers at once may work but has not been extensively tested.
YMMV.

#### Failover

Scheduler failover may be triggered by either the following events:

- loss of leadership when running in HA mode (`--ha`).
- the leading scheduler process receives a USR1 signal.

It is currently possible signal failover to a single, non-HA scheduler process.
In this case, if there are problems launching a replacement scheduler process then the cluster may be without a scheduler until another is manually started.

#### How To

##### Command Line Arguments

- `--ha` is required to enable scheduler HA and multi-scheduler leader election.
- `--km-path` or else (`--executor-path` and `--proxy-path`) should reference non-local-file URI's and must be identical across schedulers.

If you have HDFS installed on your slaves then you can specify HDFS URI locations for the binaries:

```shell
$ hdfs dfs -put -f bin/km hdfs:///km
$ ./bin/km scheduler ... --mesos-master=zk://zk1:2181,zk2:2181/mesos --ha --km-path=hdfs:///km
```

**IMPORTANT:** some command line parameters specified for the scheduler process are passed to the Kubelet-executor and so are subject to compatibility tests:

- a Mesos master will not recognize differently configured executors as being compatible, and so...
- a scheduler will refuse to accept any offer for slave resources if there are incompatible executors running on the slave.

Within the scheduler, compatibility is largely determined by comparing executor configuration hashes:
  a hash is calculated from a subset of the executor-related command line parameters provided to the scheduler process.
The command line parameters that affect the hash calculation are listed below.

- `--allow-privileged`
- `--api-servers`
- `--auth-path`
- `--cluster_*`
- `--executor_*`
- `--kubelet_*`
- `--km-path`
- `--profiling`
- `--proxy_path`


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/mesos/docs/ha.md?pixel)]()
