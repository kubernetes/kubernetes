## kube-controller-manager



### Synopsis


The Kubernetes controller manager is a daemon that embeds
the core control loops shipped with Kubernetes. In applications of robotics and
automation, a control loop is a non-terminating loop that regulates the state of
the system. In Kubernetes, a controller is a control loop that watches the shared
state of the cluster through the apiserver and makes changes attempting to move the
current state towards the desired state. Examples of controllers that ship with
Kubernetes today are the replication controller, endpoints controller, namespace
controller, and serviceaccounts controller.

```
kube-controller-manager
```

### Options

```
      --address ip                                                        The IP address to serve on (set to 0.0.0.0 for all interfaces) (default 0.0.0.0)
      --allocate-node-cidrs                                               Should CIDRs for Pods be allocated and set on the cloud provider.
      --attach-detach-reconcile-sync-period duration                      The reconciler sync wait time between volume attach detach. This duration must be larger than one second, and increasing this value from the default may allow for volumes to be mismatched with pods. (default 1m0s)
      --azure-container-registry-config string                            Path to the file container Azure container registry configuration information.
      --cloud-config string                                               The path to the cloud provider configuration file.  Empty string for no configuration file.
      --cloud-provider string                                             The provider for cloud services.  Empty string for no provider.
      --cluster-cidr string                                               CIDR Range for Pods in cluster.
      --cluster-name string                                               The instance prefix for the cluster (default "kubernetes")
      --cluster-signing-cert-file string                                  Filename containing a PEM-encoded X509 CA certificate used to issue cluster-scoped certificates (default "/etc/kubernetes/ca/ca.pem")
      --cluster-signing-key-file string                                   Filename containing a PEM-encoded RSA or ECDSA private key used to sign cluster-scoped certificates (default "/etc/kubernetes/ca/ca.key")
      --concurrent-deployment-syncs int32                                 The number of deployment objects that are allowed to sync concurrently. Larger number = more responsive deployments, but more CPU (and network) load (default 5)
      --concurrent-endpoint-syncs int32                                   The number of endpoint syncing operations that will be done concurrently. Larger number = faster endpoint updating, but more CPU (and network) load (default 5)
      --concurrent-gc-syncs int32                                         The number of garbage collector workers that are allowed to sync concurrently. (default 20)
      --concurrent-namespace-syncs int32                                  The number of namespace objects that are allowed to sync concurrently. Larger number = more responsive namespace termination, but more CPU (and network) load (default 2)
      --concurrent-replicaset-syncs int32                                 The number of replica sets that are allowed to sync concurrently. Larger number = more responsive replica management, but more CPU (and network) load (default 5)
      --concurrent-resource-quota-syncs int32                             The number of resource quotas that are allowed to sync concurrently. Larger number = more responsive quota management, but more CPU (and network) load (default 5)
      --concurrent-service-syncs int32                                    The number of services that are allowed to sync concurrently. Larger number = more responsive service management, but more CPU (and network) load (default 1)
      --concurrent-serviceaccount-token-syncs int32                       The number of service account token objects that are allowed to sync concurrently. Larger number = more responsive token generation, but more CPU (and network) load (default 5)
      --concurrent_rc_syncs int32                                         The number of replication controllers that are allowed to sync concurrently. Larger number = more responsive replica management, but more CPU (and network) load (default 5)
      --configure-cloud-routes                                            Should CIDRs allocated by allocate-node-cidrs be configured on the cloud provider. (default true)
      --controller-start-interval duration                                Interval between starting controller managers.
      --controllers stringSlice                                           A list of controllers to enable.  '*' enables all on-by-default controllers, 'foo' enables the controller named 'foo', '-foo' disables the controller named 'foo'.
All controllers: bootstrapsigner, certificatesigningrequests, cronjob, daemonset, deployment, disruption, endpoint, garbagecollector, horizontalpodautoscaling, job, namespace, podgc, replicaset, replicationcontroller, resourcequota, serviceaccount, statefuleset, tokencleaner, ttl
Disabled-by-default controllers: bootstrapsigner, tokencleaner (default [*])
      --daemonset-lookup-cache-size int32                                 The the size of lookup cache for daemonsets. Larger number = more responsive daemonsets, but more MEM load. (default 1024)
      --deployment-controller-sync-period duration                        Period for syncing the deployments. (default 30s)
      --disable-attach-detach-reconcile-sync                              Disable volume attach detach reconciler sync. Disabling this may cause volumes to be mismatched with pods. Use wisely.
      --enable-dynamic-provisioning                                       Enable dynamic provisioning for environments that support it. (default true)
      --enable-garbage-collector                                          Enables the generic garbage collector. MUST be synced with the corresponding flag of the kube-apiserver. (default true)
      --enable-hostpath-provisioner                                       Enable HostPath PV provisioning when running without a cloud provider. This allows testing and development of provisioning features.  HostPath provisioning is not supported in any way, won't work in a multi-node cluster, and should not be used for anything other than testing or development.
      --enable-taint-manager                                              WARNING: Beta feature. If set to true enables NoExecute Taints and will evict all not-tolerating Pod running on Nodes tainted with this kind of Taints. (default true)
      --feature-gates mapStringBool                                       A set of key=value pairs that describe feature gates for alpha/experimental features. Options are:
AffinityInAnnotations=true|false (ALPHA - default=false)
AllAlpha=true|false (ALPHA - default=false)
AllowExtTrafficLocalEndpoints=true|false (BETA - default=true)
AppArmor=true|false (BETA - default=true)
DynamicKubeletConfig=true|false (ALPHA - default=false)
DynamicVolumeProvisioning=true|false (ALPHA - default=true)
ExperimentalCriticalPodAnnotation=true|false (ALPHA - default=false)
ExperimentalHostUserNamespaceDefaulting=true|false (BETA - default=false)
StreamingProxyRedirects=true|false (BETA - default=true)
      --flex-volume-plugin-dir string                                     Full path of the directory in which the flex volume plugin should search for additional third party volume plugins. (default "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/")
      --google-json-key string                                            The Google Cloud Platform Service Account JSON Key to use for authentication.
      --horizontal-pod-autoscaler-sync-period duration                    The period for syncing the number of pods in horizontal pod autoscaler. (default 30s)
      --insecure-experimental-approve-all-kubelet-csrs-for-group string   The group for which the controller-manager will auto approve all CSRs for kubelet client certificates.
      --kube-api-burst int32                                              Burst to use while talking with kubernetes apiserver (default 30)
      --kube-api-content-type string                                      Content type of requests sent to apiserver. (default "application/vnd.kubernetes.protobuf")
      --kube-api-qps float32                                              QPS to use while talking with kubernetes apiserver (default 20)
      --kubeconfig string                                                 Path to kubeconfig file with authorization and master location information.
      --large-cluster-size-threshold int32                                Number of nodes from which NodeController treats the cluster as large for the eviction logic purposes. --secondary-node-eviction-rate is implicitly overridden to 0 for clusters this size or smaller. (default 50)
      --leader-elect                                                      Start a leader election client and gain leadership before executing the main loop. Enable this when running replicated components for high availability. (default true)
      --leader-elect-lease-duration duration                              The duration that non-leader candidates will wait after observing a leadership renewal until attempting to acquire leadership of a led but unrenewed leader slot. This is effectively the maximum duration that a leader can be stopped before it is replaced by another candidate. This is only applicable if leader election is enabled. (default 15s)
      --leader-elect-renew-deadline duration                              The interval between attempts by the acting master to renew a leadership slot before it stops leading. This must be less than or equal to the lease duration. This is only applicable if leader election is enabled. (default 10s)
      --leader-elect-retry-period duration                                The duration the clients should wait between attempting acquisition and renewal of a leadership. This is only applicable if leader election is enabled. (default 2s)
      --master string                                                     The address of the Kubernetes API server (overrides any value in kubeconfig)
      --min-resync-period duration                                        The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod (default 12h0m0s)
      --namespace-sync-period duration                                    The period for syncing namespace life-cycle updates (default 5m0s)
      --node-cidr-mask-size int32                                         Mask size for node cidr in cluster. (default 24)
      --node-eviction-rate float32                                        Number of nodes per second on which pods are deleted in case of node failure when a zone is healthy (see --unhealthy-zone-threshold for definition of healthy/unhealthy). Zone refers to entire cluster in non-multizone clusters. (default 0.1)
      --node-monitor-grace-period duration                                Amount of time which we allow running Node to be unresponsive before marking it unhealthy. Must be N times more than kubelet's nodeStatusUpdateFrequency, where N means number of retries allowed for kubelet to post node status. (default 40s)
      --node-monitor-period duration                                      The period for syncing NodeStatus in NodeController. (default 5s)
      --node-startup-grace-period duration                                Amount of time which we allow starting Node to be unresponsive before marking it unhealthy. (default 1m0s)
      --pod-eviction-timeout duration                                     The grace period for deleting pods on failed nodes. (default 5m0s)
      --port int32                                                        The port that the controller-manager's http service runs on (default 10252)
      --profiling                                                         Enable profiling via web interface host:port/debug/pprof/ (default true)
      --pv-recycler-increment-timeout-nfs int32                           the increment of time added per Gi to ActiveDeadlineSeconds for an NFS scrubber pod (default 30)
      --pv-recycler-minimum-timeout-hostpath int32                        The minimum ActiveDeadlineSeconds to use for a HostPath Recycler pod.  This is for development and testing only and will not work in a multi-node cluster. (default 60)
      --pv-recycler-minimum-timeout-nfs int32                             The minimum ActiveDeadlineSeconds to use for an NFS Recycler pod (default 300)
      --pv-recycler-pod-template-filepath-hostpath string                 The file path to a pod definition used as a template for HostPath persistent volume recycling. This is for development and testing only and will not work in a multi-node cluster.
      --pv-recycler-pod-template-filepath-nfs string                      The file path to a pod definition used as a template for NFS persistent volume recycling
      --pv-recycler-timeout-increment-hostpath int32                      the increment of time added per Gi to ActiveDeadlineSeconds for a HostPath scrubber pod.  This is for development and testing only and will not work in a multi-node cluster. (default 30)
      --pvclaimbinder-sync-period duration                                The period for syncing persistent volumes and persistent volume claims (default 15s)
      --replicaset-lookup-cache-size int32                                The the size of lookup cache for replicatsets. Larger number = more responsive replica management, but more MEM load. (default 4096)
      --replication-controller-lookup-cache-size int32                    The the size of lookup cache for replication controllers. Larger number = more responsive replica management, but more MEM load. (default 4096)
      --resource-quota-sync-period duration                               The period for syncing quota usage status in the system (default 5m0s)
      --root-ca-file string                                               If set, this root certificate authority will be included in service account's token secret. This must be a valid PEM-encoded CA bundle.
      --route-reconciliation-period duration                              The period for reconciling routes created for Nodes by cloud provider. (default 10s)
      --secondary-node-eviction-rate float32                              Number of nodes per second on which pods are deleted in case of node failure when a zone is unhealthy (see --unhealthy-zone-threshold for definition of healthy/unhealthy). Zone refers to entire cluster in non-multizone clusters. This value is implicitly overridden to 0 if the cluster size is smaller than --large-cluster-size-threshold. (default 0.01)
      --service-account-private-key-file string                           Filename containing a PEM-encoded private RSA or ECDSA key used to sign service account tokens.
      --service-cluster-ip-range string                                   CIDR Range for Services in cluster.
      --service-sync-period duration                                      The period for syncing services with their external load balancers (default 5m0s)
      --terminated-pod-gc-threshold int32                                 Number of terminated pods that can exist before the terminated pod garbage collector starts deleting terminated pods. If <= 0, the terminated pod garbage collector is disabled. (default 12500)
      --unhealthy-zone-threshold float32                                  Fraction of Nodes in a zone which needs to be not Ready (minimum 3) for zone to be treated as unhealthy.  (default 0.55)
      --use-service-account-credentials                                   If true, use individual service account credentials for each controller.
```

###### Auto generated by spf13/cobra on 21-Feb-2017
