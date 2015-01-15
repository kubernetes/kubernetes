% KUBERNETES(1) kubernetes User Manuals
% Scott Collier
% October 2014
# NAME
kube-apiserver \- Provides the API for kubernetes orchestration.

# SYNOPSIS
**kube-apiserver** [OPTIONS]

# DESCRIPTION

The **kubernetes** API server validates and configures data for 3 types of objects: pods, services, and replicationControllers. Beyond just servicing REST operations, the API Server does two other things as well: 1. Schedules pods to worker nodes. Right now the scheduler is very simple. 2. Synchronize pod information (where they are, what ports they are exposing) with the service configuration.

The the kube-apiserver several options.

# OPTIONS
**--address**=""
	The address on the local server to listen to. Default 127.0.0.1

**--allow_privileged**=""
	If true, allow privileged containers.

**--alsologtostderr**=
	log to standard error as well as files. Default is false.

**--api_prefix**="/api"
	The prefix for API requests on the server. Default '/api'

**--cloud_config**=""
	The path to the cloud provider configuration file. Empty string for no configuration file.

**--cloud_provider**=""
	The provider for cloud services. Empty string for no provider.

**--cors_allowed_origins**=[]
	List of allowed origins for CORS, comma separated. An allowed origin can be a regular expression to support subdomain matching. If this list is empty CORS will not be enabled.

**--etcd_servers**=[]
	List of etcd servers to watch (http://ip:port), comma separated

**--health_check_minions**=
	If true, health check minions and filter unhealthy ones. Default true.

**--log_backtrace_at=**:0
	when logging hits line file:N, emit a stack trace

**--log_dir**=""
	If non-empty, write log files in this directory

**--log_flush_frequency**=5s
	Maximum number of seconds between log flushes. Default is 5 seconds.

**--logtostderr**=
	log to standard error instead of files. Default is false.

**--kubelet_port**=10250
	The port at which kubelet will be listening on the minions. Default is 10250.

**--port**=8080
	The port to listen on. Default is 8080.

**--stderrthreshold**=0
	logs at or above this threshold go to stderr. Default is 0.

**--storage_version**=""
	The version to store resources with. Defaults to server preferred.

**--v**=0
	Log level for V logs.

**--version**=false
	Print version information and quit. Default is false.

**--vmodule**=
	comma-separated list of pattern=N settings for file-filtered logging

# EXAMPLES
```
/usr/bin/kube-apiserver --logtostderr=true --v=0 --etcd_servers=http://127.0.0.1:4001 --address=0.0.0.0 --port=8080 --kubelet_port=10250 --allow_privileged=false
```
# HISTORY
October 2014, Originally compiled by Scott Collier (scollier at redhat dot com) based
 on the kubernetes source material and internal work.
