## kube-scheduler



### Synopsis


The Kubernetes scheduler is a policy-rich, topology-aware,
workload-specific function that significantly impacts availability, performance,
and capacity. The scheduler needs to take into account individual and collective
resource requirements, quality of service requirements, hardware/software/policy
constraints, affinity and anti-affinity specifications, data locality, inter-workload
interference, deadlines, and so on. Workload-specific requirements will be exposed
through the API as necessary.

```
kube-scheduler
```

### Options

```
      --address string                           The IP address to serve on (set to 0.0.0.0 for all interfaces) (default "0.0.0.0")
      --algorithm-provider string                The scheduling algorithm provider to use, one of: ClusterAutoscalerProvider | DefaultProvider (default "DefaultProvider")
      --azure-container-registry-config string   Path to the file container Azure container registry configuration information.
      --contention-profiling                     Enable lock contention profiling, if profiling is enabled
      --feature-gates mapStringBool              A set of key=value pairs that describe feature gates for alpha/experimental features. Options are:
Accelerators=true|false (ALPHA - default=false)
AffinityInAnnotations=true|false (ALPHA - default=false)
AllAlpha=true|false (ALPHA - default=false)
AllowExtTrafficLocalEndpoints=true|false (BETA - default=true)
AppArmor=true|false (BETA - default=true)
DynamicKubeletConfig=true|false (ALPHA - default=false)
DynamicVolumeProvisioning=true|false (ALPHA - default=true)
ExperimentalCriticalPodAnnotation=true|false (ALPHA - default=false)
ExperimentalHostUserNamespaceDefaulting=true|false (BETA - default=false)
StreamingProxyRedirects=true|false (BETA - default=true)
TaintBasedEvictions=true|false (ALPHA - default=false)
      --google-json-key string                   The Google Cloud Platform Service Account JSON Key to use for authentication.
      --hard-pod-affinity-symmetric-weight int   RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule corresponding to every RequiredDuringScheduling affinity rule. --hard-pod-affinity-symmetric-weight represents the weight of implicit PreferredDuringScheduling affinity rule. (default 1)
      --kube-api-burst int32                     Burst to use while talking with kubernetes apiserver (default 100)
      --kube-api-content-type string             Content type of requests sent to apiserver. (default "application/vnd.kubernetes.protobuf")
      --kube-api-qps float32                     QPS to use while talking with kubernetes apiserver (default 50)
      --kubeconfig string                        Path to kubeconfig file with authorization and master location information.
      --leader-elect                             Start a leader election client and gain leadership before executing the main loop. Enable this when running replicated components for high availability. (default true)
      --leader-elect-lease-duration duration     The duration that non-leader candidates will wait after observing a leadership renewal until attempting to acquire leadership of a led but unrenewed leader slot. This is effectively the maximum duration that a leader can be stopped before it is replaced by another candidate. This is only applicable if leader election is enabled. (default 15s)
      --leader-elect-renew-deadline duration     The interval between attempts by the acting master to renew a leadership slot before it stops leading. This must be less than or equal to the lease duration. This is only applicable if leader election is enabled. (default 10s)
      --leader-elect-retry-period duration       The duration the clients should wait between attempting acquisition and renewal of a leadership. This is only applicable if leader election is enabled. (default 2s)
      --lock-object-name string                  Define the name of the lock object. (default "kube-scheduler")
      --lock-object-namespace string             Define the namespace of the lock object. (default "kube-system")
      --master string                            The address of the Kubernetes API server (overrides any value in kubeconfig)
      --policy-config-file string                File with scheduler policy configuration. This file is used if policy ConfigMap is not provided or --use-legacy-policy-config==true
      --policy-configmap string                  Name of the ConfigMap object that contains scheduler's policy configuration. It must exist in the system namespace before scheduler initialization if --use-legacy-policy-config==false. The config must be provided as the value of an element in 'Data' map with the key='policy.cfg'
      --policy-configmap-namespace string        The namespace where policy ConfigMap is located. The system namespace will be used if this is not provided or is empty. (default "kube-system")
      --port int32                               The port that the scheduler's http service runs on (default 10251)
      --profiling                                Enable profiling via web interface host:port/debug/pprof/ (default true)
      --scheduler-name string                    Name of the scheduler, used to select which pods will be processed by this scheduler, based on pod's "spec.SchedulerName". (default "default-scheduler")
      --use-legacy-policy-config                 When set to true, scheduler will ignore policy ConfigMap and uses policy config file
```

###### Auto generated by spf13/cobra on 8-May-2017
