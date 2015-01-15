% KUBERNETES(1) kubernetes User Manuals
% Scott Collier
% October 2014
# NAME
kube-controller-manager \- Enforces kubernetes services.

# SYNOPSIS
**kube-controller-manager** [OPTIONS]

# DESCRIPTION

The **kubernetes** controller manager is really a service that is layered on top of the simple pod API. To enforce this layering, the logic for the replicationController is actually broken out into another server. This server watches etcd for changes to replicationController objects and then uses the public Kubernetes API to implement the replication algorithm.

The kube-controller-manager has several options.

# OPTIONS
**--address**=""
	The address on the local server to listen to. Default 127.0.0.1.

**--allow_privileged**="false"
	If true, allow privileged containers.

**--address=**"127.0.0.1"
	The address to serve from.

**--alsologtostderr**=false
	log to standard error as well as files.

**--api_version**=""
	The API version to use when talking to the server.

**--cloud_config**=""
	The path to the cloud provider configuration file. Empty string for no configuration file.

**--cloud_provider**=""
	The provider for cloud services. Empty string for no provider.

**--minion_regexp**=""
	If non empty, and --cloud_provider is specified, a regular expression for matching minion VMs.

**--insecure_skip_tls_verify**=false
	If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.

**--log_backtrace_at**=:0
	when logging hits line file:N, emit a stack trace.

**--log_dir**=""
	If non-empty, write log files in this directory.

**--log_flush_frequency**=5s
	Maximum number of seconds between log flushes.

**--logtostderr**=false
	log to standard error instead of files.

**--machines**=[]
    List of machines to schedule onto, comma separated.

**--sync_nodes**=true
        If true, and --cloud_provider is specified, sync nodes from the cloud provider. Default true.

**--master**=""
	The address of the Kubernetes API server.

**--node_sync_peroid**=10s
    The period for syncing nodes from cloudprovider.

**--port**=10252
	The port that the controller-manager's http service runs on.

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
/usr/bin/kube-controller-manager --logtostderr=true --v=0 --master=127.0.0.1:8080
```
# HISTORY
October 2014, Originally compiled by Scott Collier (scollier at redhat dot com) based
 on the kubernetes source material and internal work.
