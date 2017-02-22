## kubelet



### Synopsis


The kubelet is the primary "node agent" that runs on each
node. The kubelet works in terms of a PodSpec. A PodSpec is a YAML or JSON object
that describes a pod. The kubelet takes a set of PodSpecs that are provided through
various mechanisms (primarily through the apiserver) and ensures that the containers
described in those PodSpecs are running and healthy. The kubelet doesn't manage
containers which were not created by Kubernetes.

Other than from an PodSpec from the apiserver, there are three ways that a container
manifest can be provided to the Kubelet.

File: Path passed as a flag on the command line. This file is rechecked every 20
seconds (configurable with a flag).

HTTP endpoint: HTTP endpoint passed as a parameter on the command line. This endpoint
is checked every 20 seconds (also configurable with a flag).

HTTP server: The kubelet can also listen for HTTP and respond to a simple API
(underspec'd currently) to submit a new manifest.

```
kubelet
```

### Options

```
      --address ip                                              The IP address for the Kubelet to serve on (set to 0.0.0.0 for all interfaces) (default 0.0.0.0)
      --allow-privileged                                        If true, allow containers to request privileged mode. [default=false]
      --anonymous-auth                                          Enables anonymous requests to the Kubelet server. Requests that are not rejected by another authentication method are treated as anonymous requests. Anonymous requests have a username of system:anonymous, and a group name of system:unauthenticated. (default true)
      --authentication-token-webhook                            Use the TokenReview API to determine authentication for bearer tokens.
      --authentication-token-webhook-cache-ttl duration         The duration to cache responses from the webhook token authenticator. (default 2m0s)
      --authorization-mode string                               Authorization mode for Kubelet server. Valid options are AlwaysAllow or Webhook. Webhook mode uses the SubjectAccessReview API to determine authorization. (default "AlwaysAllow")
      --authorization-webhook-cache-authorized-ttl duration     The duration to cache 'authorized' responses from the webhook authorizer. (default 5m0s)
      --authorization-webhook-cache-unauthorized-ttl duration   The duration to cache 'unauthorized' responses from the webhook authorizer. (default 30s)
      --azure-container-registry-config string                  Path to the file container Azure container registry configuration information.
      --cadvisor-port int32                                     The port of the localhost cAdvisor endpoint (default 4194)
      --cert-dir string                                         The directory where the TLS certs are located (by default /var/run/kubernetes). If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored. (default "/var/run/kubernetes")
      --cgroup-driver string                                    Driver that the kubelet uses to manipulate cgroups on the host.  Possible values: 'cgroupfs', 'systemd' (default "cgroupfs")
      --cgroup-root string                                      Optional root cgroup to use for pods. This is handled by the container runtime on a best effort basis. Default: '', which means use the container runtime default.
      --cgroups-per-qos                                         Enable creation of QoS cgroup hierarchy, if true top level QoS and pod cgroups are created.
      --chaos-chance float                                      If > 0.0, introduce random client errors and latency. Intended for testing. [default=0.0]
      --client-ca-file string                                   If set, any request presenting a client certificate signed by one of the authorities in the client-ca-file is authenticated with an identity corresponding to the CommonName of the client certificate.
      --cloud-config string                                     The path to the cloud provider configuration file.  Empty string for no configuration file.
      --cloud-provider string                                   The provider for cloud services. By default, kubelet will attempt to auto-detect the cloud provider. Specify empty string for running with no cloud provider. [default=auto-detect] (default "auto-detect")
      --cluster-dns stringSlice                                 Comma-separated list of DNS server IP address.  This value is used for containers DNS server in case of Pods with "dnsPolicy=ClusterFirst". Note: all DNS servers appearing in the list MUST serve the same set of records otherwise name resolution within the cluster may not work correctly. There is no guarantee as to which DNS server may be contacted for name resolution.
      --cluster-domain string                                   Domain for this cluster.  If set, kubelet will configure all containers to search this domain in addition to the host's search domains
      --cni-bin-dir string                                      <Warning: Alpha feature> The full path of the directory in which to search for CNI plugin binaries. Default: /opt/cni/bin
      --cni-conf-dir string                                     <Warning: Alpha feature> The full path of the directory in which to search for CNI config files. Default: /etc/cni/net.d
      --container-runtime string                                The container runtime to use. Possible values: 'docker', 'rkt'. Default: 'docker'. (default "docker")
      --container-runtime-endpoint string                       [Experimental] The unix socket endpoint of remote runtime service. The endpoint is used only when CRI integration is enabled (--enable-cri)
      --containerized                                           Experimental support for running kubelet in a container.  Intended for testing. [default=false]
      --cpu-cfs-quota                                           Enable CPU CFS quota enforcement for containers that specify CPU limits (default true)
      --docker-endpoint string                                  Use this for the docker endpoint to communicate with (default "unix:///var/run/docker.sock")
      --enable-controller-attach-detach                         Enables the Attach/Detach controller to manage attachment/detachment of volumes scheduled to this node, and disables kubelet from executing any attach/detach operations (default true)
      --enable-custom-metrics                                   Support for gathering custom metrics.
      --enable-debugging-handlers                               Enables server endpoints for log collection and local running of containers and commands (default true)
      --enable-server                                           Enable the Kubelet's server (default true)
      --event-burst int32                                       Maximum size of a bursty event records, temporarily allows event records to burst to this number, while still not exceeding event-qps. Only used if --event-qps > 0 (default 10)
      --event-qps int32                                         If > 0, limit event creations per second to this value. If 0, unlimited. (default 5)
      --eviction-hard string                                    A set of eviction thresholds (e.g. memory.available<1Gi) that if met would trigger a pod eviction. (default "memory.available<100Mi")
      --eviction-max-pod-grace-period int32                     Maximum allowed grace period (in seconds) to use when terminating pods in response to a soft eviction threshold being met.  If negative, defer to pod specified value.
      --eviction-minimum-reclaim string                         A set of minimum reclaims (e.g. imagefs.available=2Gi) that describes the minimum amount of resource the kubelet will reclaim when performing a pod eviction if that resource is under pressure.
      --eviction-pressure-transition-period duration            Duration for which the kubelet has to wait before transitioning out of an eviction pressure condition. (default 5m0s)
      --eviction-soft string                                    A set of eviction thresholds (e.g. memory.available<1.5Gi) that if met over a corresponding grace period would trigger a pod eviction.
      --eviction-soft-grace-period string                       A set of eviction grace periods (e.g. memory.available=1m30s) that correspond to how long a soft eviction threshold must hold before triggering a pod eviction.
      --exit-on-lock-contention                                 Whether kubelet should exit upon lock-file contention.
      --experimental-allowed-unsafe-sysctls stringSlice         Comma-separated whitelist of unsafe sysctls or unsafe sysctl patterns (ending in *). Use these at your own risk.
      --experimental-bootstrap-kubeconfig string                <Warning: Experimental feature> Path to a kubeconfig file that will be used to get client certificate for kubelet. If the file specified by --kubeconfig does not exist, the bootstrap kubeconfig is used to request a client certificate from the API server. On success, a kubeconfig file referencing the generated key and obtained certificate is written to the path specified by --kubeconfig. The certificate and key file will be stored in the directory pointed by --cert-dir.
      --experimental-check-node-capabilities-before-mount       [Experimental] if set true, the kubelet will check the underlying node for required componenets (binaries, etc.) before performing the mount
      --experimental-fail-swap-on                               Makes the Kubelet fail to start if swap is enabled on the node. This is a temporary opton to maintain legacy behavior, failing due to swap enabled will happen by default in v1.6.
      --experimental-kernel-memcg-notification                  If enabled, the kubelet will integrate with the kernel memcg notification to determine if memory eviction thresholds are crossed rather than polling.
      --experimental-mounter-path string                        [Experimental] Path of mounter binary. Leave empty to use the default mount.
      --experimental-nvidia-gpus int32                          Number of NVIDIA GPU devices on this node. Only 0 (default) and 1 are currently supported.
      --feature-gates string                                    A set of key=value pairs that describe feature gates for alpha/experimental features. Options are:
AffinityInAnnotations=true|false (ALPHA - default=false)
AllAlpha=true|false (ALPHA - default=false)
AllowExtTrafficLocalEndpoints=true|false (BETA - default=true)
AppArmor=true|false (BETA - default=true)
DynamicKubeletConfig=true|false (ALPHA - default=false)
DynamicVolumeProvisioning=true|false (ALPHA - default=true)
ExperimentalCriticalPodAnnotation=true|false (ALPHA - default=false)
ExperimentalHostUserNamespaceDefaulting=true|false (BETA - default=false)
StreamingProxyRedirects=true|false (BETA - default=true)
      --file-check-frequency duration                           Duration between checking config files for new data (default 20s)
      --google-json-key string                                  The Google Cloud Platform Service Account JSON Key to use for authentication.
      --hairpin-mode string                                     How should the kubelet setup hairpin NAT. This allows endpoints of a Service to loadbalance back to themselves if they should try to access their own Service. Valid values are "promiscuous-bridge", "hairpin-veth" and "none". (default "promiscuous-bridge")
      --healthz-bind-address ip                                 The IP address for the healthz server to serve on, defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces) (default 127.0.0.1)
      --healthz-port int32                                      The port of the localhost healthz endpoint (default 10248)
      --host-ipc-sources stringSlice                            Comma-separated list of sources from which the Kubelet allows pods to use the host ipc namespace. [default="*"] (default [*])
      --host-network-sources stringSlice                        Comma-separated list of sources from which the Kubelet allows pods to use of host network. [default="*"] (default [*])
      --host-pid-sources stringSlice                            Comma-separated list of sources from which the Kubelet allows pods to use the host pid namespace. [default="*"] (default [*])
      --hostname-override string                                If non-empty, will use this string as identification instead of the actual hostname.
      --http-check-frequency duration                           Duration between checking http for new data (default 20s)
      --image-gc-high-threshold int32                           The percent of disk usage after which image garbage collection is always run. Default: 90% (default 90)
      --image-gc-low-threshold int32                            The percent of disk usage before which image garbage collection is never run. Lowest disk usage to garbage collect to. Default: 80% (default 80)
      --image-pull-progress-deadline duration                   If no pulling progress is made before this deadline, the image pulling will be cancelled. Default: 1m0s. (default 1m0s)
      --image-service-endpoint string                           [Experimental] The unix socket endpoint of remote image service. If not specified, it will be the same with container-runtime-endpoint by default. The endpoint is used only when CRI integration is enabled (--enable-cri)
      --iptables-drop-bit int32                                 The bit of the fwmark space to mark packets for dropping. Must be within the range [0, 31]. (default 15)
      --iptables-masquerade-bit int32                           The bit of the fwmark space to mark packets for SNAT. Must be within the range [0, 31]. Please match this parameter with corresponding parameter in kube-proxy. (default 14)
      --keep-terminated-pod-volumes                             Keep terminated pod volumes mounted to the node after the pod terminates.  Can be useful for debugging volume related issues.
      --kube-api-burst int32                                    Burst to use while talking with kubernetes apiserver (default 10)
      --kube-api-content-type string                            Content type of requests sent to apiserver. (default "application/vnd.kubernetes.protobuf")
      --kube-api-qps int32                                      QPS to use while talking with kubernetes apiserver (default 5)
      --kube-reserved mapStringString                           A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs that describe resources reserved for kubernetes system components. Currently only cpu and memory are supported. See http://kubernetes.io/docs/user-guide/compute-resources for more detail. [default=none]
      --kubeconfig string                                       Path to a kubeconfig file, specifying how to connect to the API server. --api-servers will be used for the location unless --require-kubeconfig is set. (default "/var/lib/kubelet/kubeconfig")
      --kubelet-cgroups string                                  Optional absolute name of cgroups to create and run the Kubelet in.
      --lock-file string                                        <Warning: Alpha feature> The path to file for kubelet to use as a lock file.
      --low-diskspace-threshold-mb int32                        The absolute free disk space, in MB, to maintain. When disk space falls below this threshold, new pods would be rejected. Default: 256 (default 256)
      --make-iptables-util-chains                               If true, kubelet will ensure iptables utility rules are present on host. (default true)
      --manifest-url string                                     URL for accessing the container manifest
      --manifest-url-header string                              HTTP header to use when accessing the manifest URL, with the key separated from the value with a ':', as in 'key:value'
      --master-service-namespace string                         The namespace from which the kubernetes master services should be injected into pods (default "default")
      --max-open-files int                                      Number of files that can be opened by Kubelet process. [default=1000000] (default 1000000)
      --max-pods int32                                          Number of Pods that can run on this Kubelet. (default 110)
      --minimum-image-ttl-duration duration                     Minimum age for an unused image before it is garbage collected.  Examples: '300ms', '10s' or '2h45m'. Default: '2m' (default 2m0s)
      --network-plugin string                                   <Warning: Alpha feature> The name of the network plugin to be invoked for various events in kubelet/pod lifecycle
      --network-plugin-dir string                               <Warning: Alpha feature> The full path of the directory in which to search for network plugins or CNI config
      --network-plugin-mtu int32                                <Warning: Alpha feature> The MTU to be passed to the network plugin, to override the default. Set to 0 to use the default 1460 MTU.
      --node-ip string                                          IP address of the node. If set, kubelet will use this IP address for the node
      --node-labels mapStringString                             <Warning: Alpha feature> Labels to add when registering the node in the cluster.  Labels must be key=value pairs separated by ','.
      --node-status-update-frequency duration                   Specifies how often kubelet posts node status to master. Note: be cautious when changing the constant, it must work with nodeMonitorGracePeriod in nodecontroller. Default: 10s (default 10s)
      --non-masquerade-cidr string                              Traffic to IPs outside this range will use IP masquerade. (default "10.0.0.0/8")
      --oom-score-adj int32                                     The oom-score-adj value for kubelet process. Values must be within the range [-1000, 1000] (default -999)
      --outofdisk-transition-frequency duration                 Duration for which the kubelet has to wait before transitioning out of out-of-disk node condition status. Default: 5m0s (default 5m0s)
      --pod-cidr string                                         The CIDR to use for pod IP addresses, only used in standalone mode.  In cluster mode, this is obtained from the master.
      --pod-infra-container-image string                        The image whose network/ipc namespaces containers in each pod will use. (default "gcr.io/google_containers/pause-amd64:3.0")
      --pod-manifest-path string                                Path to to the directory containing pod manifest files to run, or the path to a single pod manifest file.
      --pods-per-core int32                                     Number of Pods per core that can run on this Kubelet. The total number of Pods on this Kubelet cannot exceed max-pods, so max-pods will be used if this calculation results in a larger number of Pods allowed on the Kubelet. A value of 0 disables this limit.
      --port int32                                              The port for the Kubelet to serve on. (default 10250)
      --protect-kernel-defaults                                 Default kubelet behaviour for kernel tuning. If set, kubelet errors if any of kernel tunables is different than kubelet defaults.
      --read-only-port int32                                    The read-only port for the Kubelet to serve on with no authentication/authorization (set to 0 to disable) (default 10255)
      --really-crash-for-testing                                If true, when panics occur crash. Intended for testing.
      --register-node                                           Register the node with the apiserver (defaults to true if --api-servers is set) (default true)
      --register-with-taints []api.Taint                        Register the node with the given list of taints (comma seperated "<key>=<value>:<effect>"). No-op if register-node is false.
      --registry-burst int32                                    Maximum size of a bursty pulls, temporarily allows pulls to burst to this number, while still not exceeding registry-qps.  Only used if --registry-qps > 0 (default 10)
      --registry-qps int32                                      If > 0, limit registry pull QPS to this value.  If 0, unlimited. [default=5.0] (default 5)
      --require-kubeconfig                                      If true the Kubelet will exit if there are configuration errors, and will ignore the value of --api-servers in favor of the server defined in the kubeconfig file.
      --resolv-conf string                                      Resolver configuration file used as the basis for the container DNS resolution configuration. (default "/etc/resolv.conf")
      --rkt-api-endpoint string                                 The endpoint of the rkt API service to communicate with. Only used if --container-runtime='rkt'. (default "localhost:15441")
      --rkt-path string                                         Path of rkt binary. Leave empty to use the first rkt in $PATH.  Only used if --container-runtime='rkt'.
      --root-dir string                                         Directory path for managing kubelet files (volume mounts,etc). (default "/var/lib/kubelet")
      --runonce                                                 If true, exit after spawning pods from local manifests or remote urls. Exclusive with --api-servers, and --enable-server
      --runtime-cgroups string                                  Optional absolute name of cgroups to create and run the runtime in.
      --runtime-request-timeout duration                        Timeout of all runtime requests except long running request - pull, logs, exec and attach. When timeout exceeded, kubelet will cancel the request, throw out an error and retry later. Default: 2m0s (default 2m0s)
      --seccomp-profile-root string                             Directory path for seccomp profiles. (default "/var/lib/kubelet/seccomp")
      --serialize-image-pulls                                   Pull images one at a time. We recommend *not* changing the default value on nodes that run docker daemon with version < 1.9 or an Aufs storage backend. Issue #10959 has more details. [default=true] (default true)
      --streaming-connection-idle-timeout duration              Maximum time a streaming connection can be idle before the connection is automatically closed. 0 indicates no timeout. Example: '5m' (default 4h0m0s)
      --sync-frequency duration                                 Max period between synchronizing running containers and config (default 1m0s)
      --system-cgroups /                                        Optional absolute name of cgroups in which to place all non-kernel processes that are not already inside a cgroup under /. Empty for no container. Rolling back the flag requires a reboot. (Default: "").
      --system-reserved mapStringString                         A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs that describe resources reserved for non-kubernetes components. Currently only cpu and memory are supported. See http://kubernetes.io/docs/user-guide/compute-resources for more detail. [default=none]
      --tls-cert-file string                                    File containing x509 Certificate for HTTPS.  (CA cert, if any, concatenated after server cert). If --tls-cert-file and --tls-private-key-file are not provided, a self-signed certificate and key are generated for the public address and saved to the directory passed to --cert-dir.
      --tls-private-key-file string                             File containing x509 private key matching --tls-cert-file.
      --volume-plugin-dir string                                <Warning: Alpha feature> The full path of the directory in which to search for additional third party volume plugins (default "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/")
      --volume-stats-agg-period duration                        Specifies interval for kubelet to calculate and cache the volume disk usage for all pods and volumes.  To disable volume calculations, set to 0.  Default: '1m' (default 1m0s)
```

###### Auto generated by spf13/cobra on 21-Feb-2017
