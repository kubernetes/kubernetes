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

**--api_version=**""
	The API version to use when talking to the server

**--bindaddress**="0.0.0.0"
	The address for the proxy server to serve on (set to 0.0.0.0 or "" for all interfaces)

**--etcd_servers**=[]
	List of etcd servers to watch (http://ip:port), comma separated (optional)

**--insecure_skip_tls_verify**=false
	If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.

**--log_backtrace_at**=:0
	when logging hits line file:N, emit a stack trace

**--log_dir**=""
	If non-empty, write log files in this directory

**--log_flush_frequency**=5s
	Maximum number of seconds between log flushes

**--logtostderr**=false
	log to standard error instead of files

**--master**=""
	The address of the Kubernetes API server

**--stderrthreshold**=0
	logs at or above this threshold go to stderr

**--v**=0
	log level for V logs

**--version**=false
	Print version information and quit

**--vmodule**=
	comma-separated list of pattern=N settings for file-filtered logging


# EXAMPLES
```
/usr/bin/kube-proxy --logtostderr=true --v=0 --etcd_servers=http://127.0.0.1:4001
```
# HISTORY
October 2014, Originally compiled by Scott Collier (scollier at redhat dot com) based
 on the kubernetes source material and internal work.
