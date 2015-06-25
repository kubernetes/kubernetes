% KUBERNETES(1) kubernetes User Manuals
% Scott Collier
% October 2014
# NAME
kube-controller-manager \- Enforces kubernetes services.

# SYNOPSIS
**kube-controller-manager** [OPTIONS]

# DESCRIPTION

The **kubernetes** controller manager is really a service that is layered on top of the simple pod API. To enforce this layering, the logic for the replicationcontroller is actually broken out into another server. This server watches etcd for changes to replicationcontroller objects and then uses the public Kubernetes API to implement the replication algorithm.

The kube-controller-manager has several options.

# OPTIONS
**--address**=127.0.0.1
	The IP address to serve on (set to 0.0.0.0 for all interfaces)

**--allocate-node-cidrs**=false
	Should CIDRs for Pods be allocated and set on the cloud provider.

**--alsologtostderr**=false
	log to standard error as well as files

**--cloud-config**=""
	The path to the cloud provider configuration file.  Empty string for no configuration file.

**--cloud-provider**=""
	The provider for cloud services.  Empty string for no provider.

**--cluster-cidr**=<nil>
	CIDR Range for Pods in cluster.

**--concurrent-endpoint-syncs**=5
	The number of endpoint syncing operations that will be done concurrently. Larger number = faster endpoint updating, but more CPU (and network) load

**--concurrent_rc_syncs**=5
	The number of replication controllers that are allowed to sync concurrently. Larger number = more reponsive replica management, but more CPU (and network) load

**--deleting-pods-burst**=10
	Number of nodes on which pods are bursty deleted in case of node failure. For more details look into RateLimiter.

**--deleting-pods-qps**=0.1
	Number of nodes per second on which pods are deleted in case of node failure.

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

**--machines**=[]
	List of machines to schedule onto, comma separated.

**--master**=""
	The address of the Kubernetes API server (overrides any value in kubeconfig)

**--minion-regexp**=""
	If non empty, and --cloud-provider is specified, a regular expression for matching minion VMs.

**--namespace-sync-period**=5m0s
	The period for syncing namespace life-cycle updates

**--node-memory**=3Gi
	The amount of memory (in bytes) provisioned on each node

**--node-milli-cpu**=1000
	The amount of MilliCPU provisioned on each node

**--node-monitor-grace-period**=40s
	Amount of time which we allow running Node to be unresponsive before marking it unhealty. Must be N times more than kubelet's nodeStatusUpdateFrequency, where N means number of retries allowed for kubelet to post node status.

**--node-monitor-period**=5s
	The period for syncing NodeStatus in NodeController.

**--node-startup-grace-period**=1m0s
	Amount of time which we allow starting Node to be unresponsive before marking it unhealty.

**--node-sync-period**=10s
	The period for syncing nodes from cloudprovider. Longer periods will result in fewer calls to cloud provider, but may delay addition of new nodes to cluster.

**--pod-eviction-timeout**=5m0s
	The grace peroid for deleting pods on failed nodes.

**--port**=10252
	The port that the controller-manager's http service runs on

**--profiling**=true
	Enable profiling via web interface host:port/debug/pprof/

**--pvclaimbinder-sync-period**=10s
	The period for syncing persistent volumes and persistent volume claims

**--register-retry-count**=10
	The number of retries for initial node registration.  Retry interval equals node-sync-period.

**--resource-quota-sync-period**=10s
	The period for syncing quota usage status in the system

**--service-account-private-key-file**=""
	Filename containing a PEM-encoded private RSA key used to sign service account tokens.

**--stderrthreshold**=2
	logs at or above this threshold go to stderr

**--sync-nodes**=true
	If true, and --cloud-provider is specified, sync nodes from the cloud provider. Default true.

**--v**=0
	log level for V logs

**--version**=false
	Print version information and quit

**--vmodule**=
	comma-separated list of pattern=N settings for file-filtered logging

# EXAMPLES
```
/usr/bin/kube-controller-manager --logtostderr=true --v=0 --master=127.0.0.1:8080 --machines=127.0.0.1 
```

# HISTORY
October 2014, Originally compiled by Scott Collier (scollier at redhat dot com) based
 on the kubernetes source material and internal work.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/man/kube-controller-manager.1.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/man/kube-controller-manager.1.md?pixel)]()
