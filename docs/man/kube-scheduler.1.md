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
**--address=**"127.0.0.1"
	The address to serve from.

**--alsologtostderr=**false
	log to standard error as well as files.

**--api_version=**""
	The API version to use when talking to the server.

**--insecure_skip_tls_verify**=false
	If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.

**--log_backtrace_at=**:0
	when logging hits line file:N, emit a stack trace.

**--log_dir=**""
	If non-empty, write log files in this directory.

**--log_flush_frequency**=5s
	Maximum number of seconds between log flushes.

**--logtostderr**=false
	log to standard error instead of files.

**--master=**""
	The address of the Kubernetes API server.

**--port=**10251
	The port that the scheduler's http service runs on.

**--stderrthreshold**=0
	logs at or above this threshold go to stderr.

**--v**=0
	log level for V logs.

**--version**=false
	Print version information and quit.

**--vmodule**=
	comma-separated list of pattern=N settings for file-filtered logging.

# EXAMPLES
```
/usr/bin/kube-scheduler --logtostderr=true --v=0 --master=127.0.0.1:8080
```
# HISTORY
October 2014, Originally compiled by Scott Collier (scollier@redhat.com) based
 on the kubernetes source material and internal work.
