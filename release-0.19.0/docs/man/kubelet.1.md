% KUBERNETES(1) kubernetes User Manuals
% Scott Collier
% October 2014
# NAME
kubelet \- Processes a container manifest so the containers are launched according to how they are described.

# SYNOPSIS
**kubelet** [OPTIONS]

# DESCRIPTION

The **kubernetes** kubelet runs on each node. The Kubelet works in terms of a container manifest. A container manifest is a YAML or JSON file that describes a pod. The Kubelet takes a set of manifests that are provided in various mechanisms and ensures that the containers described in those manifests are started and continue running.

There are 3 ways that a container manifest can be provided to the Kubelet:

    File: Path passed as a flag on the command line. This file is rechecked every 20 seconds (configurable with a flag).
    HTTP endpoint: HTTP endpoint passed as a parameter on the command line. This endpoint is checked every 20 seconds (also configurable with a flag).
    HTTP server: The kubelet can also listen for HTTP and respond to a simple API (underspec'd currently) to submit a new manifest.

# OPTIONS
**--address**=0.0.0.0
	The IP address for the info server to serve on (set to 0.0.0.0 for all interfaces)

**--allow_dynamic_housekeeping**=true
	Whether to allow the housekeeping interval to be dynamic

**--allow-privileged**=false
	If true, allow containers to request privileged mode. [default=false]

**--alsologtostderr**=false
	log to standard error as well as files

**--api-servers**=[]
	List of Kubernetes API servers for publishing events, and reading pods and services. (ip:port), comma separated.

**--boot_id_file**=/proc/sys/kernel/random/boot_id
	Comma-separated list of files to check for boot-id. Use the first one that exists.

**--cadvisor-port**=4194
	The port of the localhost cAdvisor endpoint

**--cert-dir**="/var/run/kubernetes"
	The directory where the TLS certs are located (by default /var/run/kubernetes). If --tls_cert_file and --tls_private_key_file are provided, this flag will be ignored.

**--cgroup_root**=""
	Optional root cgroup to use for pods. This is handled by the container runtime on a best effort basis. Default: '', which means use the container runtime default.

**--cloud-config**=""
	The path to the cloud provider configuration file.  Empty string for no configuration file.

**--cloud-provider**=""
	The provider for cloud services.  Empty string for no provider.

**--cluster-dns**=<nil>
	IP address for a cluster DNS server.  If set, kubelet will configure all containers to use this for DNS resolution in addition to the host's DNS servers

**--cluster-domain**=""
	Domain for this cluster.  If set, kubelet will configure all containers to search this domain in addition to the host's search domains

**--config**=""
	Path to the config file or directory of files

**--configure-cbr0**=false
	If true, kubelet will configure cbr0 based on Node.Spec.PodCIDR.

**--container_hints**=/etc/cadvisor/container_hints.json
	location of the container hints file

**--container_runtime**="docker"
	The container runtime to use. Possible values: 'docker', 'rkt'. Default: 'docker'.

**--docker**=unix:///var/run/docker.sock
	docker endpoint

**--docker-daemon-container**="/docker-daemon"
	Optional resource-only container in which to place the Docker Daemon. Empty for no container (Default: /docker-daemon).

**--docker-endpoint**=""
	If non-empty, use this for the docker endpoint to communicate with

**--docker_only**=false
	Only report docker containers in addition to root stats

**--docker_root**=/var/lib/docker
	Absolute path to the Docker state root directory (default: /var/lib/docker)

**--docker_run**=/var/run/docker
	Absolute path to the Docker run directory (default: /var/run/docker)

**--enable-debugging-handlers**=true
	Enables server endpoints for log collection and local running of containers and commands

**--enable_load_reader**=false
	Whether to enable cpu load reader

**--enable-server**=true
	Enable the info server

**--event_storage_age_limit**=default=24h
	Max length of time for which to store events (per type). Value is a comma separated list of key values, where the keys are event types (e.g.: creation, oom) or "default" and the value is a duration. Default is applied to all non-specified event types

**--event_storage_event_limit**=default=100000
	Max number of events to store (per type). Value is a comma separated list of key values, where the keys are event types (e.g.: creation, oom) or "default" and the value is an integer. Default is applied to all non-specified event types

**--file-check-frequency**=20s
	Duration between checking config files for new data

**--global_housekeeping_interval**=1m0s
	Interval between global housekeepings

**--google-json-key**=""
	The Google Cloud Platform Service Account JSON Key to use for authentication.

**--healthz-bind-address**=127.0.0.1
	The IP address for the healthz server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)

**--healthz-port**=10248
	The port of the localhost healthz endpoint

**--host-network-sources**="file"
	Comma-separated list of sources from which the Kubelet allows pods to use of host network. For all sources use "*" [default="file"]

**--hostname-override**=""
	If non-empty, will use this string as identification instead of the actual hostname.

**--housekeeping_interval**=1s
	Interval between container housekeepings

**--http-check-frequency**=20s
	Duration between checking http for new data

**--image-gc-high-threshold**=90
	The percent of disk usage after which image garbage collection is always run. Default: 90%%

**--image-gc-low-threshold**=80
	The percent of disk usage before which image garbage collection is never run. Lowest disk usage to garbage collect to. Default: 80%%

**--kubeconfig**=/var/lib/kubelet/kubeconfig
	Path to a kubeconfig file, specifying how to authenticate to API server (the master location is set by the api-servers flag).

**--log_backtrace_at**=:0
	when logging hits line file:N, emit a stack trace

**--log_cadvisor_usage**=false
	Whether to log the usage of the cAdvisor container

**--log_dir**=
	If non-empty, write log files in this directory

**--log_flush_frequency**=5s
	Maximum number of seconds between log flushes

**--logtostderr**=true
	log to standard error instead of files

**--low-diskspace-threshold-mb**=256
	The absolute free disk space, in MB, to maintain. When disk space falls below this threshold, new pods would be rejected. Default: 256

**--machine_id_file**=/etc/machine-id,/var/lib/dbus/machine-id
	Comma-separated list of files to check for machine-id. Use the first one that exists.

**--manifest-url**=""
	URL for accessing the container manifest

**--master-service-namespace**="default"
	The namespace from which the kubernetes master services should be injected into pods

**--max_housekeeping_interval**=1m0s
	Largest interval to allow between container housekeepings

**--max_pods**=100
	Number of Pods that can run on this Kubelet.

**--maximum-dead-containers**=100
	Maximum number of old instances of a containers to retain globally.  Each container takes up some disk space.  Default: 100.

**--maximum-dead-containers-per-container**=5
	Maximum number of old instances of a container to retain per container.  Each container takes up some disk space.  Default: 5.

**--minimum-container-ttl-duration**=1m0s
	Minimum age for a finished container before it is garbage collected.  Examples: '300ms', '10s' or '2h45m'

**--network-plugin**=""
	The name of the network plugin to be invoked for various events in kubelet/pod lifecycle

**--node-status-update-frequency**=10s
	Specifies how often kubelet posts node status to master. Note: be cautious when changing the constant, it must work with nodeMonitorGracePeriod in nodecontroller. Default: 10s

**--oom-score-adj**=-900
	The oom_score_adj value for kubelet process. Values must be within the range [-1000, 1000]

**--pod-infra-container-image**="gcr.io/google_containers/pause:0.8.0"
	The image whose network/ipc namespaces containers in each pod will use.

**--port**=10250
	The port for the info server to serve on

**--read-only-port**=10255
	The read-only port for the info server to serve on (set to 0 to disable)

**--registry-burst**=10
	Maximum size of a bursty pulls, temporarily allows pulls to burst to this number, while still not exceeding registry_qps.  Only used if --registry_qps > 0

**--registry-qps**=0
	If > 0, limit registry pull QPS to this value.  If 0, unlimited. [default=0.0]

**--resource-container**="/kubelet"
	Absolute name of the resource-only container to create and run the Kubelet in (Default: /kubelet).

**--root-dir**="/var/lib/kubelet"
	Directory path for managing kubelet files (volume mounts,etc).

**--runonce**=false
	If true, exit after spawning pods from local manifests or remote urls. Exclusive with --api_servers, and --enable-server

**--stderrthreshold**=2
	logs at or above this threshold go to stderr

**--streaming-connection-idle-timeout**=0
	Maximum time a streaming connection can be idle before the connection is automatically closed.  Example: '5m'

**--sync-frequency**=10s
	Max period between synchronizing running containers and config

**--tls-cert-file**=""
	File /gmrvcontaining x509 Certificate for HTTPS.  (CA cert, if any, concatenated after server cert). If --tls_cert_file and --tls_private_key_file are not provided, a self-signed certificate and key are generated for the public address and saved to the directory passed to --cert_dir.

**--tls-private-key-file**=""
	File containing x509 private key matching --tls_cert_file.

**--v**=0
	log level for V logs

**--version**=false
	Print version information and quit

**--vmodule**=
	comma-separated list of pattern=N settings for file-filtered logging

# EXAMPLES
```
/usr/bin/kubelet --logtostderr=true --v=0 --api_servers=http://127.0.0.1:8080 --address=127.0.0.1 --port=10250 --hostname_override=127.0.0.1 --allow-privileged=false
```

# HISTORY
October 2014, Originally compiled by Scott Collier (scollier at redhat dot com) based
 on the kubernetes source material and internal work.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/man/kubelet.1.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/man/kubelet.1.md?pixel)]()
