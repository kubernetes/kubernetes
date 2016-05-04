/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package componentconfig

import "k8s.io/kubernetes/pkg/api/unversioned"

type KubeProxyConfiguration struct {
	unversioned.TypeMeta

	// bindAddress is the IP address for the proxy server to serve on (set to 0.0.0.0
	// for all interfaces)
	BindAddress string `json:"bindAddress"`
	// clusterCIDR is the CIDR range of the pods in the cluster. It is used to
	// bridge traffic coming from outside of the cluster. If not provided,
	// no off-cluster bridging will be performed.
	ClusterCIDR string `json:"clusterCIDR"`
	// healthzBindAddress is the IP address for the health check server to serve on,
	// defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)
	HealthzBindAddress string `json:"healthzBindAddress"`
	// healthzPort is the port to bind the health check server. Use 0 to disable.
	HealthzPort int32 `json:"healthzPort"`
	// hostnameOverride, if non-empty, will be used as the identity instead of the actual hostname.
	HostnameOverride string `json:"hostnameOverride"`
	// iptablesMasqueradeBit is the bit of the iptables fwmark space to use for SNAT if using
	// the pure iptables proxy mode. Values must be within the range [0, 31].
	IPTablesMasqueradeBit *int32 `json:"iptablesMasqueradeBit"`
	// iptablesSyncPeriod is the period that iptables rules are refreshed (e.g. '5s', '1m',
	// '2h22m').  Must be greater than 0.
	IPTablesSyncPeriod unversioned.Duration `json:"iptablesSyncPeriodSeconds"`
	// kubeconfigPath is the path to the kubeconfig file with authorization information (the
	// master location is set by the master flag).
	KubeconfigPath string `json:"kubeconfigPath"`
	// masqueradeAll tells kube-proxy to SNAT everything if using the pure iptables proxy mode.
	MasqueradeAll bool `json:"masqueradeAll"`
	// master is the address of the Kubernetes API server (overrides any value in kubeconfig)
	Master string `json:"master"`
	// oomScoreAdj is the oom-score-adj value for kube-proxy process. Values must be within
	// the range [-1000, 1000]
	OOMScoreAdj *int32 `json:"oomScoreAdj"`
	// mode specifies which proxy mode to use.
	Mode ProxyMode `json:"mode"`
	// portRange is the range of host ports (beginPort-endPort, inclusive) that may be consumed
	// in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.
	PortRange string `json:"portRange"`
	// resourceContainer is the absolute name of the resource-only container to create and run
	// the Kube-proxy in (Default: /kube-proxy).
	ResourceContainer string `json:"kubeletCgroups"`
	// udpIdleTimeout is how long an idle UDP connection will be kept open (e.g. '250ms', '2s').
	// Must be greater than 0. Only applicable for proxyMode=userspace.
	UDPIdleTimeout unversioned.Duration `json:"udpTimeoutMilliseconds"`
	// conntrackMax is the maximum number of NAT connections to track (0 to leave as-is)")
	ConntrackMax int32 `json:"conntrackMax"`
	// conntrackTCPEstablishedTimeout is how long an idle UDP connection will be kept open
	// (e.g. '250ms', '2s').  Must be greater than 0. Only applicable for proxyMode is Userspace
	ConntrackTCPEstablishedTimeout unversioned.Duration `json:"conntrackTCPEstablishedTimeout"`
}

// Currently two modes of proxying are available: 'userspace' (older, stable) or 'iptables'
// (newer, faster). If blank, look at the Node object on the Kubernetes API and respect the
// 'net.experimental.kubernetes.io/proxy-mode' annotation if provided.  Otherwise use the
// best-available proxy (currently iptables, but may change in future versions).  If the
// iptables proxy is selected, regardless of how, but the system's kernel or iptables
// versions are insufficient, this always falls back to the userspace proxy.
type ProxyMode string

const (
	ProxyModeUserspace ProxyMode = "userspace"
	ProxyModeIPTables  ProxyMode = "iptables"
)

// HairpinMode denotes how the kubelet should configure networking to handle
// hairpin packets.
type HairpinMode string

// Enum settings for different ways to handle hairpin packets.
const (
	// Set the hairpin flag on the veth of containers in the respective
	// container runtime.
	HairpinVeth = "hairpin-veth"
	// Make the container bridge promiscuous. This will force it to accept
	// hairpin packets, even if the flag isn't set on ports of the bridge.
	PromiscuousBridge = "promiscuous-bridge"
	// Neither of the above. If the kubelet is started in this hairpin mode
	// and kube-proxy is running in iptables mode, hairpin packets will be
	// dropped by the container bridge.
	HairpinNone = "none"
)

// TODO: curate the ordering and structure of this config object
type KubeletConfiguration struct {
	// config is the path to the config file or directory of files
	Config string `json:"config"`
	// syncFrequency is the max period between synchronizing running
	// containers and config
	SyncFrequency unversioned.Duration `json:"syncFrequency"`
	// fileCheckFrequency is the duration between checking config files for
	// new data
	FileCheckFrequency unversioned.Duration `json:"fileCheckFrequency"`
	// httpCheckFrequency is the duration between checking http for new data
	HTTPCheckFrequency unversioned.Duration `json:"httpCheckFrequency"`
	// manifestURL is the URL for accessing the container manifest
	ManifestURL string `json:"manifestURL"`
	// manifestURLHeader is the HTTP header to use when accessing the manifest
	// URL, with the key separated from the value with a ':', as in 'key:value'
	ManifestURLHeader string `json:"manifestURLHeader"`
	// enableServer enables the Kubelet's server
	EnableServer bool `json:"enableServer"`
	// address is the IP address for the Kubelet to serve on (set to 0.0.0.0
	// for all interfaces)
	Address string `json:"address"`
	// port is the port for the Kubelet to serve on.
	Port uint `json:"port"`
	// readOnlyPort is the read-only port for the Kubelet to serve on with
	// no authentication/authorization (set to 0 to disable)
	ReadOnlyPort uint `json:"readOnlyPort"`
	// tLSCertFile is the file containing x509 Certificate for HTTPS.  (CA cert,
	// if any, concatenated after server cert). If tlsCertFile and
	// tlsPrivateKeyFile are not provided, a self-signed certificate
	// and key are generated for the public address and saved to the directory
	// passed to certDir.
	TLSCertFile string `json:"tLSCertFile"`
	// tLSPrivateKeyFile is the ile containing x509 private key matching
	// tlsCertFile.
	TLSPrivateKeyFile string `json:"tLSPrivateKeyFile"`
	// certDirectory is the directory where the TLS certs are located (by
	// default /var/run/kubernetes). If tlsCertFile and tlsPrivateKeyFile
	// are provided, this flag will be ignored.
	CertDirectory string `json:"certDirectory"`
	// hostnameOverride is the hostname used to identify the kubelet instead
	// of the actual hostname.
	HostnameOverride string `json:"hostnameOverride"`
	// podInfraContainerImage is the image whose network/ipc namespaces
	// containers in each pod will use.
	PodInfraContainerImage string `json:"podInfraContainerImage"`
	// dockerEndpoint is the path to the docker endpoint to communicate with.
	DockerEndpoint string `json:"dockerEndpoint"`
	// rootDirectory is the directory path to place kubelet files (volume
	// mounts,etc).
	RootDirectory string `json:"rootDirectory"`
	// allowPrivileged enables containers to request privileged mode.
	// Defaults to false.
	AllowPrivileged bool `json:"allowPrivileged"`
	// hostNetworkSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use of host network. Defaults to "*".
	HostNetworkSources string `json:"hostNetworkSources"`
	// hostPIDSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use the host pid namespace. Defaults to "*".
	HostPIDSources string `json:"hostPIDSources"`
	// hostIPCSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use the host ipc namespace. Defaults to "*".
	HostIPCSources string `json:"hostIPCSources"`
	// registryPullQPS is the limit of registry pulls per second. If 0,
	// unlimited. Set to 0 for no limit. Defaults to 5.0.
	RegistryPullQPS float64 `json:"registryPullQPS"`
	// registryBurst is the maximum size of a bursty pulls, temporarily allows
	// pulls to burst to this number, while still not exceeding registryQps.
	// Only used if registryQps > 0.
	RegistryBurst int32 `json:"registryBurst"`
	// eventRecordQPS is the maximum event creations per second. If 0, there
	// is no limit enforced.
	EventRecordQPS float32 `json:"eventRecordQPS"`
	// eventBurst is the maximum size of a bursty event records, temporarily
	// allows event records to burst to this number, while still not exceeding
	// event-qps. Only used if eventQps > 0
	EventBurst int32 `json:"eventBurst"`
	// enableDebuggingHandlers enables server endpoints for log collection
	// and local running of containers and commands
	EnableDebuggingHandlers bool `json:"enableDebuggingHandlers"`
	// minimumGCAge is the minimum age for a finished container before it is
	// garbage collected.
	MinimumGCAge unversioned.Duration `json:"minimumGCAge"`
	// maxPerPodContainerCount is the maximum number of old instances to
	// retain per container. Each container takes up some disk space.
	MaxPerPodContainerCount int32 `json:"maxPerPodContainerCount"`
	// maxContainerCount is the maximum number of old instances of containers
	// to retain globally. Each container takes up some disk space.
	MaxContainerCount int32 `json:"maxContainerCount"`
	// cAdvisorPort is the port of the localhost cAdvisor endpoint
	CAdvisorPort uint `json:"cAdvisorPort"`
	// healthzPort is the port of the localhost healthz endpoint
	HealthzPort int32 `json:"healthzPort"`
	// healthzBindAddress is the IP address for the healthz server to serve
	// on.
	HealthzBindAddress string `json:"healthzBindAddress"`
	// oomScoreAdj is The oom-score-adj value for kubelet process. Values
	// must be within the range [-1000, 1000].
	OOMScoreAdj int32 `json:"oomScoreAdj"`
	// registerNode enables automatic registration with the apiserver.
	RegisterNode bool `json:"registerNode"`
	// clusterDomain is the DNS domain for this cluster. If set, kubelet will
	// configure all containers to search this domain in addition to the
	// host's search domains.
	ClusterDomain string `json:"clusterDomain"`
	// masterServiceNamespace is The namespace from which the kubernetes
	// master services should be injected into pods.
	MasterServiceNamespace string `json:"masterServiceNamespace"`
	// clusterDNS is the IP address for a cluster DNS server.  If set, kubelet
	// will configure all containers to use this for DNS resolution in
	// addition to the host's DNS servers
	ClusterDNS string `json:"clusterDNS"`
	// streamingConnectionIdleTimeout is the maximum time a streaming connection
	// can be idle before the connection is automatically closed.
	StreamingConnectionIdleTimeout unversioned.Duration `json:"streamingConnectionIdleTimeout"`
	// nodeStatusUpdateFrequency is the frequency that kubelet posts node
	// status to master. Note: be cautious when changing the constant, it
	// must work with nodeMonitorGracePeriod in nodecontroller.
	NodeStatusUpdateFrequency unversioned.Duration `json:"nodeStatusUpdateFrequency"`
	// minimumGCAge is the minimum age for a unused image before it is
	// garbage collected.
	ImageMinimumGCAge unversioned.Duration `json:"imageMinimumGCAge"`
	// imageGCHighThresholdPercent is the percent of disk usage after which
	// image garbage collection is always run.
	ImageGCHighThresholdPercent int32 `json:"imageGCHighThresholdPercent"`
	// imageGCLowThresholdPercent is the percent of disk usage before which
	// image garbage collection is never run. Lowest disk usage to garbage
	// collect to.
	ImageGCLowThresholdPercent int32 `json:"imageGCLowThresholdPercent"`
	// lowDiskSpaceThresholdMB is the absolute free disk space, in MB, to
	// maintain. When disk space falls below this threshold, new pods would
	// be rejected.
	LowDiskSpaceThresholdMB int32 `json:"lowDiskSpaceThresholdMB"`
	// How frequently to calculate and cache volume disk usage for all pods
	VolumeStatsAggPeriod unversioned.Duration `json:"volumeStatsAggPeriod"`
	// networkPluginName is the name of the network plugin to be invoked for
	// various events in kubelet/pod lifecycle
	NetworkPluginName string `json:"networkPluginName"`
	// networkPluginDir is the full path of the directory in which to search
	// for network plugins
	NetworkPluginDir string `json:"networkPluginDir"`
	// volumePluginDir is the full path of the directory in which to search
	// for additional third party volume plugins
	VolumePluginDir string `json:"volumePluginDir"`
	// cloudProvider is the provider for cloud services.
	CloudProvider string `json:"cloudProvider,omitempty"`
	// cloudConfigFile is the path to the cloud provider configuration file.
	CloudConfigFile string `json:"cloudConfigFile,omitempty"`
	// KubeletCgroups is the absolute name of cgroups to isolate the kubelet in.
	KubeletCgroups string `json:"kubeletCgroups,omitempty"`
	// Cgroups that container runtime is expected to be isolated in.
	RuntimeCgroups string `json:"runtimeCgroups,omitempty"`
	// SystemCgroups is absolute name of cgroups in which to place
	// all non-kernel processes that are not already in a container. Empty
	// for no container. Rolling back the flag requires a reboot.
	SystemCgroups string `json:"systemContainer,omitempty"`
	// cgroupRoot is the root cgroup to use for pods. This is handled by the
	// container runtime on a best effort basis.
	CgroupRoot string `json:"cgroupRoot,omitempty"`
	// containerRuntime is the container runtime to use.
	ContainerRuntime string `json:"containerRuntime"`
	// rktPath is the path of rkt binary. Leave empty to use the first rkt in
	// $PATH.
	RktPath string `json:"rktPath,omitempty"`
	// rktApiEndpoint is the endpoint of the rkt API service to communicate with.
	RktAPIEndpoint string `json:"rktAPIEndpoint,omitempty"`
	// rktStage1Image is the image to use as stage1. Local paths and
	// http/https URLs are supported.
	RktStage1Image string `json:"rktStage1Image,omitempty"`
	// lockFilePath is the path that kubelet will use to as a lock file.
	// It uses this file as a lock to synchronize with other kubelet processes
	// that may be running.
	LockFilePath string `json:"lockFilePath"`
	// configureCBR0 enables the kublet to configure cbr0 based on
	// Node.Spec.PodCIDR.
	ConfigureCBR0 bool `json:"configureCbr0"`
	// How should the kubelet configure the container bridge for hairpin packets.
	// Setting this flag allows endpoints in a Service to loadbalance back to
	// themselves if they should try to access their own Service. Values:
	//   "promiscuous-bridge": make the container bridge promiscuous.
	//   "hairpin-veth":       set the hairpin flag on container veth interfaces.
	//   "none":               do nothing.
	// Setting --configure-cbr0 to false implies that to achieve hairpin NAT
	// one must set --hairpin-mode=veth-flag, because bridge assumes the
	// existence of a container bridge named cbr0.
	HairpinMode string `json:"hairpinMode"`
	// The node has babysitter process monitoring docker and kubelet.
	BabysitDaemons bool `json:"babysitDaemons"`
	// maxPods is the number of pods that can run on this Kubelet.
	MaxPods int32 `json:"maxPods"`
	// dockerExecHandlerName is the handler to use when executing a command
	// in a container. Valid values are 'native' and 'nsenter'. Defaults to
	// 'native'.
	DockerExecHandlerName string `json:"dockerExecHandlerName"`
	// The CIDR to use for pod IP addresses, only used in standalone mode.
	// In cluster mode, this is obtained from the master.
	PodCIDR string `json:"podCIDR"`
	// ResolverConfig is the resolver configuration file used as the basis
	// for the container DNS resolution configuration."), []
	ResolverConfig string `json:"resolvConf"`
	// cpuCFSQuota is Enable CPU CFS quota enforcement for containers that
	// specify CPU limits
	CPUCFSQuota bool `json:"cpuCFSQuota"`
	// containerized should be set to true if kubelet is running in a container.
	Containerized bool `json:"containerized"`
	// maxOpenFiles is Number of files that can be opened by Kubelet process.
	MaxOpenFiles uint64 `json:"maxOpenFiles"`
	// reconcileCIDR is Reconcile node CIDR with the CIDR specified by the
	// API server. No-op if register-node or configure-cbr0 is false.
	ReconcileCIDR bool `json:"reconcileCIDR"`
	// registerSchedulable tells the kubelet to register the node as
	// schedulable. No-op if register-node is false.
	RegisterSchedulable bool `json:"registerSchedulable"`
	// contentType is contentType of requests sent to apiserver.
	ContentType string `json:"contentType"`
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver
	KubeAPIQPS float32 `json:"kubeAPIQPS"`
	// kubeAPIBurst is the burst to allow while talking with kubernetes
	// apiserver
	KubeAPIBurst int32 `json:"kubeAPIBurst"`
	// serializeImagePulls when enabled, tells the Kubelet to pull images one
	// at a time. We recommend *not* changing the default value on nodes that
	// run docker daemon with version  < 1.9 or an Aufs storage backend.
	// Issue #10959 has more details.
	SerializeImagePulls bool `json:"serializeImagePulls"`
	// experimentalFlannelOverlay enables experimental support for starting the
	// kubelet with the default overlay network (flannel). Assumes flanneld
	// is already running in client mode.
	ExperimentalFlannelOverlay bool `json:"experimentalFlannelOverlay"`
	// outOfDiskTransitionFrequency is duration for which the kubelet has to
	// wait before transitioning out of out-of-disk node condition status.
	OutOfDiskTransitionFrequency unversioned.Duration `json:"outOfDiskTransitionFrequency,omitempty"`
	// nodeIP is IP address of the node. If set, kubelet will use this IP
	// address for the node.
	NodeIP string `json:"nodeIP,omitempty"`
	// nodeLabels to add when registering the node in the cluster.
	NodeLabels map[string]string `json:"nodeLabels"`
	// nonMasqueradeCIDR configures masquerading: traffic to IPs outside this range will use IP masquerade.
	NonMasqueradeCIDR string `json:"nonMasqueradeCIDR"`
	// enable gathering custom metrics.
	EnableCustomMetrics bool `json:"enableCustomMetrics"`
	// Comma-delimited list of hard eviction expressions.  For example, 'memory.available<300Mi'.
	EvictionHard string `json:"evictionHard,omitempty"`
	// Comma-delimited list of soft eviction expressions.  For example, 'memory.available<300Mi'.
	EvictionSoft string `json:"evictionSoft,omitempty"`
	// Comma-delimeted list of grace periods for each soft eviction signal.  For example, 'memory.available=30s'.
	EvictionSoftGracePeriod string `json:"evictionSoftGracePeriod,omitempty"`
}

type KubeSchedulerConfiguration struct {
	unversioned.TypeMeta

	// port is the port that the scheduler's http service runs on.
	Port int32 `json:"port"`
	// address is the IP address to serve on.
	Address string `json:"address"`
	// algorithmProvider is the scheduling algorithm provider to use.
	AlgorithmProvider string `json:"algorithmProvider"`
	// policyConfigFile is the filepath to the scheduler policy configuration.
	PolicyConfigFile string `json:"policyConfigFile"`
	// enableProfiling enables profiling via web interface.
	EnableProfiling bool `json:"enableProfiling"`
	// contentType is contentType of requests sent to apiserver.
	ContentType string `json:"contentType"`
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver.
	KubeAPIQPS float32 `json:"kubeAPIQPS"`
	// kubeAPIBurst is the QPS burst to use while talking with kubernetes apiserver.
	KubeAPIBurst int32 `json:"kubeAPIBurst"`
	// schedulerName is name of the scheduler, used to select which pods
	// will be processed by this scheduler, based on pod's annotation with
	// key 'scheduler.alpha.kubernetes.io/name'.
	SchedulerName string `json:"schedulerName"`
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	HardPodAffinitySymmetricWeight int `json:"hardPodAffinitySymmetricWeight"`
	// Indicate the "all topologies" set for empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.
	FailureDomains string `json:"failureDomains"`
	// leaderElection defines the configuration of leader election client.
	LeaderElection LeaderElectionConfiguration `json:"leaderElection"`
}

// LeaderElectionConfiguration defines the configuration of leader election
// clients for components that can run with leader election enabled.
type LeaderElectionConfiguration struct {
	// leaderElect enables a leader election client to gain leadership
	// before executing the main loop. Enable this when running replicated
	// components for high availability.
	LeaderElect bool `json:"leaderElect"`
	// leaseDuration is the duration that non-leader candidates will wait
	// after observing a leadership renewal until attempting to acquire
	// leadership of a led but unrenewed leader slot. This is effectively the
	// maximum duration that a leader can be stopped before it is replaced
	// by another candidate. This is only applicable if leader election is
	// enabled.
	LeaseDuration unversioned.Duration `json:"leaseDuration"`
	// renewDeadline is the interval between attempts by the acting master to
	// renew a leadership slot before it stops leading. This must be less
	// than or equal to the lease duration. This is only applicable if leader
	// election is enabled.
	RenewDeadline unversioned.Duration `json:"renewDeadline"`
	// retryPeriod is the duration the clients should wait between attempting
	// acquisition and renewal of a leadership. This is only applicable if
	// leader election is enabled.
	RetryPeriod unversioned.Duration `json:"retryPeriod"`
}

type KubeControllerManagerConfiguration struct {
	unversioned.TypeMeta

	// port is the port that the controller-manager's http service runs on.
	Port int32 `json:"port"`
	// address is the IP address to serve on (set to 0.0.0.0 for all interfaces).
	Address string `json:"address"`
	// cloudProvider is the provider for cloud services.
	CloudProvider string `json:"cloudProvider"`
	// cloudConfigFile is the path to the cloud provider configuration file.
	CloudConfigFile string `json:"cloudConfigFile"`
	// concurrentEndpointSyncs is the number of endpoint syncing operations
	// that will be done concurrently. Larger number = faster endpoint updating,
	// but more CPU (and network) load.
	ConcurrentEndpointSyncs int32 `json:"concurrentEndpointSyncs"`
	// concurrentRSSyncs is the number of replica sets that are  allowed to sync
	// concurrently. Larger number = more responsive replica  management, but more
	// CPU (and network) load.
	ConcurrentRSSyncs int32 `json:"concurrentRSSyncs"`
	// concurrentRCSyncs is the number of replication controllers that are
	// allowed to sync concurrently. Larger number = more responsive replica
	// management, but more CPU (and network) load.
	ConcurrentRCSyncs int32 `json:"concurrentRCSyncs"`
	// concurrentResourceQuotaSyncs is the number of resource quotas that are
	// allowed to sync concurrently. Larger number = more responsive quota
	// management, but more CPU (and network) load.
	ConcurrentResourceQuotaSyncs int32 `json:"concurrentResourceQuotaSyncs"`
	// concurrentDeploymentSyncs is the number of deployment objects that are
	// allowed to sync concurrently. Larger number = more responsive deployments,
	// but more CPU (and network) load.
	ConcurrentDeploymentSyncs int32 `json:"concurrentDeploymentSyncs"`
	// concurrentDaemonSetSyncs is the number of daemonset objects that are
	// allowed to sync concurrently. Larger number = more responsive daemonset,
	// but more CPU (and network) load.
	ConcurrentDaemonSetSyncs int32 `json:"concurrentDaemonSetSyncs"`
	// concurrentJobSyncs is the number of job objects that are
	// allowed to sync concurrently. Larger number = more responsive jobs,
	// but more CPU (and network) load.
	ConcurrentJobSyncs int32 `json:"concurrentJobSyncs"`
	// concurrentNamespaceSyncs is the number of namespace objects that are
	// allowed to sync concurrently.
	ConcurrentNamespaceSyncs int32 `json:"concurrentNamespaceSyncs"`
	// lookupCacheSizeForRC is the size of lookup cache for replication controllers.
	// Larger number = more responsive replica management, but more MEM load.
	LookupCacheSizeForRC int32 `json:"lookupCacheSizeForRC"`
	// lookupCacheSizeForRS is the size of lookup cache for replicatsets.
	// Larger number = more responsive replica management, but more MEM load.
	LookupCacheSizeForRS int32 `json:"lookupCacheSizeForRS"`
	// lookupCacheSizeForDaemonSet is the size of lookup cache for daemonsets.
	// Larger number = more responsive daemonset, but more MEM load.
	LookupCacheSizeForDaemonSet int32 `json:"lookupCacheSizeForDaemonSet"`
	// serviceSyncPeriod is the period for syncing services with their external
	// load balancers.
	ServiceSyncPeriod unversioned.Duration `json:"serviceSyncPeriod"`
	// nodeSyncPeriod is the period for syncing nodes from cloudprovider. Longer
	// periods will result in fewer calls to cloud provider, but may delay addition
	// of new nodes to cluster.
	NodeSyncPeriod unversioned.Duration `json:"nodeSyncPeriod"`
	// resourceQuotaSyncPeriod is the period for syncing quota usage status
	// in the system.
	ResourceQuotaSyncPeriod unversioned.Duration `json:"resourceQuotaSyncPeriod"`
	// namespaceSyncPeriod is the period for syncing namespace life-cycle
	// updates.
	NamespaceSyncPeriod unversioned.Duration `json:"namespaceSyncPeriod"`
	// pvClaimBinderSyncPeriod is the period for syncing persistent volumes
	// and persistent volume claims.
	PVClaimBinderSyncPeriod unversioned.Duration `json:"pvClaimBinderSyncPeriod"`
	// minResyncPeriod is the resync period in reflectors; will be random between
	// minResyncPeriod and 2*minResyncPeriod.
	MinResyncPeriod unversioned.Duration `json:"minResyncPeriod"`
	// terminatedPodGCThreshold is the number of terminated pods that can exist
	// before the terminated pod garbage collector starts deleting terminated pods.
	// If <= 0, the terminated pod garbage collector is disabled.
	TerminatedPodGCThreshold int32 `json:"terminatedPodGCThreshold"`
	// horizontalPodAutoscalerSyncPeriod is the period for syncing the number of
	// pods in horizontal pod autoscaler.
	HorizontalPodAutoscalerSyncPeriod unversioned.Duration `json:"horizontalPodAutoscalerSyncPeriod"`
	// deploymentControllerSyncPeriod is the period for syncing the deployments.
	DeploymentControllerSyncPeriod unversioned.Duration `json:"deploymentControllerSyncPeriod"`
	// podEvictionTimeout is the grace period for deleting pods on failed nodes.
	PodEvictionTimeout unversioned.Duration `json:"podEvictionTimeout"`
	// deletingPodsQps is the number of nodes per second on which pods are deleted in
	// case of node failure.
	DeletingPodsQps float32 `json:"deletingPodsQps"`
	// deletingPodsBurst is the number of nodes on which pods are bursty deleted in
	// case of node failure. For more details look into RateLimiter.
	DeletingPodsBurst int32 `json:"deletingPodsBurst"`
	// nodeMontiorGracePeriod is the amount of time which we allow a running node to be
	// unresponsive before marking it unhealty. Must be N times more than kubelet's
	// nodeStatusUpdateFrequency, where N means number of retries allowed for kubelet
	// to post node status.
	NodeMonitorGracePeriod unversioned.Duration `json:"nodeMonitorGracePeriod"`
	// registerRetryCount is the number of retries for initial node registration.
	// Retry interval equals node-sync-period.
	RegisterRetryCount int32 `json:"registerRetryCount"`
	// nodeStartupGracePeriod is the amount of time which we allow starting a node to
	// be unresponsive before marking it unhealty.
	NodeStartupGracePeriod unversioned.Duration `json:"nodeStartupGracePeriod"`
	// nodeMonitorPeriod is the period for syncing NodeStatus in NodeController.
	NodeMonitorPeriod unversioned.Duration `json:"nodeMonitorPeriod"`
	// serviceAccountKeyFile is the filename containing a PEM-encoded private RSA key
	// used to sign service account tokens.
	ServiceAccountKeyFile string `json:"serviceAccountKeyFile"`
	// enableProfiling enables profiling via web interface host:port/debug/pprof/
	EnableProfiling bool `json:"enableProfiling"`
	// clusterName is the instance prefix for the cluster.
	ClusterName string `json:"clusterName"`
	// clusterCIDR is CIDR Range for Pods in cluster.
	ClusterCIDR string `json:"clusterCIDR"`
	// allocateNodeCIDRs enables CIDRs for Pods to be allocated and set on the
	// cloud provider.
	AllocateNodeCIDRs bool `json:"allocateNodeCIDRs"`
	// rootCAFile is the root certificate authority will be included in service
	// account's token secret. This must be a valid PEM-encoded CA bundle.
	RootCAFile string `json:"rootCAFile"`
	// contentType is contentType of requests sent to apiserver.
	ContentType string `json:"contentType"`
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver.
	KubeAPIQPS float32 `json:"kubeAPIQPS"`
	// kubeAPIBurst is the burst to use while talking with kubernetes apiserver.
	KubeAPIBurst int32 `json:"kubeAPIBurst"`
	// leaderElection defines the configuration of leader election client.
	LeaderElection LeaderElectionConfiguration `json:"leaderElection"`
	// volumeConfiguration holds configuration for volume related features.
	VolumeConfiguration VolumeConfiguration `json:"volumeConfiguration"`
	// How long to wait between starting controller managers
	ControllerStartInterval unversioned.Duration `json:"controllerStartInterval"`
}

// VolumeConfiguration contains *all* enumerated flags meant to configure all volume
// plugins. From this config, the controller-manager binary will create many instances of
// volume.VolumeConfig, each containing only the configuration needed for that plugin which
// are then passed to the appropriate plugin. The ControllerManager binary is the only part
// of the code which knows what plugins are supported and which flags correspond to each plugin.
type VolumeConfiguration struct {
	// enableHostPathProvisioning enables HostPath PV provisioning when running without a
	// cloud provider. This allows testing and development of provisioning features. HostPath
	// provisioning is not supported in any way, won't work in a multi-node cluster, and
	// should not be used for anything other than testing or development.
	EnableHostPathProvisioning bool `json:"enableHostPathProvisioning"`
	// persistentVolumeRecyclerConfiguration holds configuration for persistent volume plugins.
	PersistentVolumeRecyclerConfiguration PersistentVolumeRecyclerConfiguration `json:"persitentVolumeRecyclerConfiguration"`
}

type PersistentVolumeRecyclerConfiguration struct {
	// maximumRetry is number of retries the PV recycler will execute on failure to recycle
	// PV.
	MaximumRetry int32 `json:"maximumRetry"`
	// minimumTimeoutNFS is the minimum ActiveDeadlineSeconds to use for an NFS Recycler
	// pod.
	MinimumTimeoutNFS int32 `json:"minimumTimeoutNFS"`
	// podTemplateFilePathNFS is the file path to a pod definition used as a template for
	// NFS persistent volume recycling
	PodTemplateFilePathNFS string `json:"podTemplateFilePathNFS"`
	// incrementTimeoutNFS is the increment of time added per Gi to ActiveDeadlineSeconds
	// for an NFS scrubber pod.
	IncrementTimeoutNFS int32 `json:"incrementTimeoutNFS"`
	// podTemplateFilePathHostPath is the file path to a pod definition used as a template for
	// HostPath persistent volume recycling. This is for development and testing only and
	// will not work in a multi-node cluster.
	PodTemplateFilePathHostPath string `json:"podTemplateFilePathHostPath"`
	// minimumTimeoutHostPath is the minimum ActiveDeadlineSeconds to use for a HostPath
	// Recycler pod.  This is for development and testing only and will not work in a multi-node
	// cluster.
	MinimumTimeoutHostPath int32 `json:"minimumTimeoutHostPath"`
	// incrementTimeoutHostPath is the increment of time added per Gi to ActiveDeadlineSeconds
	// for a HostPath scrubber pod.  This is for development and testing only and will not work
	// in a multi-node cluster.
	IncrementTimeoutHostPath int32 `json:"incrementTimeoutHostPath"`
}
