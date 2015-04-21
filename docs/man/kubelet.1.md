% KUBERNETES(1) kubernetes User Manuals
% Scott Collier
% October 2014
# NAME
kubelet \- Processes a container manifest so the containers are launched according to how they are described.

# SYNOPSIS
**kubelet** [OPTIONS]

# DESCRIPTION

The **kubernetes** kubelet runs on each node. The Kubelet works in terms of a container manifest. A container manifest is a YAML or JSON file that describes a pod. The Kubelet takes a set of manifests that are provided in various mechanisms and ensures that the containers described in those manifests are started and continue running.

There are 4 ways that a container manifest can be provided to the Kubelet:

    File Path passed as a flag on the command line. This file is rechecked every 20 seconds (configurable with a flag).
    HTTP endpoint HTTP endpoint passed as a parameter on the command line. This endpoint is checked every 20 seconds (also configurable with a flag).
    etcd server The Kubelet will reach out and do a watch on an etcd server. The etcd path that is watched is /registry/hosts/$(uname -n). As this is a watch, changes are noticed and acted upon very quickly.
    HTTP server The kubelet can also listen for HTTP and respond to a simple API (underspec'd currently) to submit a new manifest.
 

# OPTIONS
**--address**="127.0.0.1"
	The address for the info server to serve on (set to 0.0.0.0 or "" for all interfaces).

**--allow_privileged**=false
	If true, allow containers to request privileged mode. [default=false].

**--alsologtostderr**=false
	log to standard error as well as files.

**--config**=""
	Path to the config file or directory of files.

**--docker_endpoint**=""
	If non-empty, use this for the docker endpoint to communicate with.

**--enable_server**=true
	Enable the info server.

**--etcd_servers**=[]
	List of etcd servers to watch (http://ip:port), comma separated.

**--file_check_frequency**=20s
	Duration between checking config files for new data.

**--hostname_override**=""
	If non-empty, will use this string as identification instead of the actual hostname.

**--http_check_frequency**=20s
	Duration between checking http for new data.

**--log_backtrace_at**=:0
	when logging hits line file:N, emit a stack trace.

**--log_dir**=""
	If non-empty, write log files in this directory.

**--log_flush_frequency**=5s
	Maximum number of seconds between log flushes.

**--logtostderr**=false
	log to standard error instead of files.

**--manifest_url**=""
	URL for accessing the container manifest.

**--pod_infra_container_image**="kubernetes/pause:latest"
	The image that pod infra containers in each pod will use.

**--port**=10250
	The port for the info server to serve on.

**--registry_burst**=10
	Maximum size of a bursty pulls, temporarily allows pulls to burst to this number, while still not exceeding registry_qps. Only used if --registry_qps > 0.

**--registry_qps**=0
	If > 0, limit registry pull QPS to this value. If 0, unlimited. [default=0.0].

**--root_dir**="/var/lib/kubelet"
	Directory path for managing kubelet files (volume mounts,etc).

**--stderrthreshold**=0
	logs at or above this threshold go to stderr.

**--sync_frequency**=10s
	Max period between synchronizing running containers and config.

**--v**=0
	log level for V logs.

**--version**=false
	Print version information and quit.

**--vmodule**=
	comma-separated list of pattern=N settings for file-filtered logging.


# EXAMPLES
```
/usr/bin/kubelet --logtostderr=true --v=0 --etcd_servers=http://127.0.0.1:4001 --address=127.0.0.1 --port=10250 --hostname_override=127.0.0.1 --allow_privileged=false
```
# HISTORY
October 2014, Originally compiled by Scott Collier (scollier at redhat dot com) based
 on the kubernetes source material and internal work.
