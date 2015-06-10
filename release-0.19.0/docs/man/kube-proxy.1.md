% KUBERNETES(1) kubernetes User Manuals
% Scott Collier
% October 2014
# NAME
kube-proxy \- Provides network proxy services.

# SYNOPSIS
**kube-proxy** [OPTIONS]

# DESCRIPTION

The **kubernetes** network proxy runs on each node. This reflects services as defined in the Kubernetes API on each node and can do simple TCP stream forwarding or round robin TCP forwarding across a set of backends. Service endpoints are currently found through Docker-links-compatible environment variables specifying ports opened by the service proxy. Currently the user must select a port to expose the service on on the proxy, as well as the container's port to target.

The kube-proxy takes several options.

# OPTIONS
**--alsologtostderr**=false
	log to standard error as well as files

**--bind-address**=0.0.0.0
	The IP address for the proxy server to serve on (set to 0.0.0.0 for all interfaces)

**--healthz-bind-address**=127.0.0.1
	The IP address for the health check server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)

**--healthz-port**=10249
	The port to bind the health check server. Use 0 to disable.

**--kubeconfig**=""
	Path to kubeconfig file with authorization information (the master location is set by the master flag).

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

**--oom-score-adj**=-899
	The oom_score_adj value for kube-proxy process. Values must be within the range [-1000, 1000]

**--resource-container**="/kube-proxy"
	Absolute name of the resource-only container to create and run the Kube-proxy in (Default: /kube-proxy).

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
/usr/bin/kube-proxy --logtostderr=true --v=0 --master=http://127.0.0.1:8080
```

# HISTORY
October 2014, Originally compiled by Scott Collier (scollier at redhat dot com) based
 on the kubernetes source material and internal work.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/man/kube-proxy.1.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/man/kube-proxy.1.md?pixel)]()
