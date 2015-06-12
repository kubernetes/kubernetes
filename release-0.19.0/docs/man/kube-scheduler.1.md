% KUBERNETES(1) kubernetes User Manuals
% Scott Collier
% October 2014
# NAME
kube-scheduler \- Schedules containers on hosts.

# SYNOPSIS
**kube-scheduler** [OPTIONS]

# DESCRIPTION

The **kubernetes** scheduler is a policy-rich, topology-aware, workload-specific function that significantly impacts availability, performance, and capacity. The scheduler needs to take into account individual and collective resource requirements, quality of service requirements, hardware/software/policy constraints, affinity and anti-affinity specifications, data locality, inter-workload interference, deadlines, and so on. Workload-specific requirements will be exposed through the API as necessary.

The kube-scheduler can take several options.

# OPTIONS
**--address**=127.0.0.1
	The IP address to serve on (set to 0.0.0.0 for all interfaces)

**--algorithm-provider**="DefaultProvider"
	The scheduling algorithm provider to use, one of: DefaultProvider

**--alsologtostderr**=false
	log to standard error as well as files

**--kubeconfig**=""
	Path to kubeconfig file with authorization and master location information.

**--log_backtrace_at**=:0
	when logging hits line file:N, emit a stack trace

**--log_dir**=
	If non-empty, write log files in this directory

**--log_flush_frequency**=5s
	Maximum number of seconds between log flushes

**--logtostderr**=true
	log to standard error instead of files

**--master**=""
	The address of the Kubernetes API server (overrides any value in kubeconfig)

**--policy-config-file**=""
	File with scheduler policy configuration

**--port**=10251
	The port that the scheduler's http service runs on

**--profiling**=true
	Enable profiling via web interface host:port/debug/pprof/

**--stderrthreshold**=2
	logs at or above this threshold go to stderr

**--v**=0
	log level for V logs

**--version**=false
	Print version information and quit

**--vmodule**=
	comma-separated list of pattern=N settings for file-filtered logging

# EXAMPLES
```
/usr/bin/kube-scheduler --logtostderr=true --v=0 --master=127.0.0.1:8080
```

# HISTORY
October 2014, Originally compiled by Scott Collier (scollier@redhat.com) based
 on the kubernetes source material and internal work.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/man/kube-scheduler.1.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/man/kube-scheduler.1.md?pixel)]()
