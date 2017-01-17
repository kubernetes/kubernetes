/*
Copyright 2015 The Kubernetes Authors.

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

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/pkg/api"
	utilconfig "k8s.io/client-go/pkg/util/config"
)

type KubeProxyConfiguration struct {
	metav1.TypeMeta

	// bindAddress is the IP address for the proxy server to serve on (set to 0.0.0.0
	// for all interfaces)
	BindAddress string
	// clusterCIDR is the CIDR range of the pods in the cluster. It is used to
	// bridge traffic coming from outside of the cluster. If not provided,
	// no off-cluster bridging will be performed.
	ClusterCIDR string
	// healthzBindAddress is the IP address for the health check server to serve on,
	// defaulting to 127.0.0.1 (set to 0.0.0.0 for all interfaces)
	HealthzBindAddress string
	// healthzPort is the port to bind the health check server. Use 0 to disable.
	HealthzPort int32
	// hostnameOverride, if non-empty, will be used as the identity instead of the actual hostname.
	HostnameOverride string
	// iptablesMasqueradeBit is the bit of the iptables fwmark space to use for SNAT if using
	// the pure iptables proxy mode. Values must be within the range [0, 31].
	IPTablesMasqueradeBit *int32
	// iptablesSyncPeriod is the period that iptables rules are refreshed (e.g. '5s', '1m',
	// '2h22m').  Must be greater than 0.
	IPTablesSyncPeriod metav1.Duration
	// iptablesMinSyncPeriod is the minimum period that iptables rules are refreshed (e.g. '5s', '1m',
	// '2h22m').
	IPTablesMinSyncPeriod metav1.Duration
	// kubeconfigPath is the path to the kubeconfig file with authorization information (the
	// master location is set by the master flag).
	KubeconfigPath string
	// masqueradeAll tells kube-proxy to SNAT everything if using the pure iptables proxy mode.
	MasqueradeAll bool
	// master is the address of the Kubernetes API server (overrides any value in kubeconfig)
	Master string
	// oomScoreAdj is the oom-score-adj value for kube-proxy process. Values must be within
	// the range [-1000, 1000]
	OOMScoreAdj *int32
	// mode specifies which proxy mode to use.
	Mode ProxyMode
	// portRange is the range of host ports (beginPort-endPort, inclusive) that may be consumed
	// in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.
	PortRange string
	// resourceContainer is the absolute name of the resource-only container to create and run
	// the Kube-proxy in (Default: /kube-proxy).
	ResourceContainer string
	// udpIdleTimeout is how long an idle UDP connection will be kept open (e.g. '250ms', '2s').
	// Must be greater than 0. Only applicable for proxyMode=userspace.
	UDPIdleTimeout metav1.Duration
	// conntrackMax is the maximum number of NAT connections to track (0 to
	// leave as-is).  This takes precedence over conntrackMaxPerCore and conntrackMin.
	ConntrackMax int32
	// conntrackMaxPerCore is the maximum number of NAT connections to track
	// per CPU core (0 to leave the limit as-is and ignore conntrackMin).
	ConntrackMaxPerCore int32
	// conntrackMin is the minimum value of connect-tracking records to allocate,
	// regardless of conntrackMaxPerCore (set conntrackMaxPerCore=0 to leave the limit as-is).
	ConntrackMin int32
	// conntrackTCPEstablishedTimeout is how long an idle TCP connection will be kept open
	// (e.g. '2s').  Must be greater than 0.
	ConntrackTCPEstablishedTimeout metav1.Duration
	// conntrackTCPCloseWaitTimeout is how long an idle conntrack entry
	// in CLOSE_WAIT state will remain in the conntrack
	// table. (e.g. '60s'). Must be greater than 0 to set.
	ConntrackTCPCloseWaitTimeout metav1.Duration
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
	metav1.TypeMeta

	// podManifestPath is the path to the directory containing pod manifests to
	// run, or the path to a single manifest file
	PodManifestPath string
	// syncFrequency is the max period between synchronizing running
	// containers and config
	SyncFrequency metav1.Duration
	// fileCheckFrequency is the duration between checking config files for
	// new data
	FileCheckFrequency metav1.Duration
	// httpCheckFrequency is the duration between checking http for new data
	HTTPCheckFrequency metav1.Duration
	// manifestURL is the URL for accessing the container manifest
	ManifestURL string
	// manifestURLHeader is the HTTP header to use when accessing the manifest
	// URL, with the key separated from the value with a ':', as in 'key:value'
	ManifestURLHeader string
	// enableServer enables the Kubelet's server
	EnableServer bool
	// address is the IP address for the Kubelet to serve on (set to 0.0.0.0
	// for all interfaces)
	Address string
	// port is the port for the Kubelet to serve on.
	Port int32
	// readOnlyPort is the read-only port for the Kubelet to serve on with
	// no authentication/authorization (set to 0 to disable)
	ReadOnlyPort int32
	// tlsCertFile is the file containing x509 Certificate for HTTPS.  (CA cert,
	// if any, concatenated after server cert). If tlsCertFile and
	// tlsPrivateKeyFile are not provided, a self-signed certificate
	// and key are generated for the public address and saved to the directory
	// passed to certDir.
	TLSCertFile string
	// tlsPrivateKeyFile is the ile containing x509 private key matching
	// tlsCertFile.
	TLSPrivateKeyFile string
	// certDirectory is the directory where the TLS certs are located (by
	// default /var/run/kubernetes). If tlsCertFile and tlsPrivateKeyFile
	// are provided, this flag will be ignored.
	CertDirectory string
	// authentication specifies how requests to the Kubelet's server are authenticated
	Authentication KubeletAuthentication
	// authorization specifies how requests to the Kubelet's server are authorized
	Authorization KubeletAuthorization
	// hostnameOverride is the hostname used to identify the kubelet instead
	// of the actual hostname.
	HostnameOverride string
	// podInfraContainerImage is the image whose network/ipc namespaces
	// containers in each pod will use.
	PodInfraContainerImage string
	// dockerEndpoint is the path to the docker endpoint to communicate with.
	DockerEndpoint string
	// rootDirectory is the directory path to place kubelet files (volume
	// mounts,etc).
	RootDirectory string
	// seccompProfileRoot is the directory path for seccomp profiles.
	SeccompProfileRoot string
	// allowPrivileged enables containers to request privileged mode.
	// Defaults to false.
	AllowPrivileged bool
	// hostNetworkSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use of host network. Defaults to "*". Valid
	// options are "file", "http", "api", and "*" (all sources).
	HostNetworkSources []string
	// hostPIDSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use the host pid namespace. Defaults to "*".
	HostPIDSources []string
	// hostIPCSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use the host ipc namespace. Defaults to "*".
	HostIPCSources []string
	// registryPullQPS is the limit of registry pulls per second. If 0,
	// unlimited. Set to 0 for no limit. Defaults to 5.0.
	RegistryPullQPS int32
	// registryBurst is the maximum size of a bursty pulls, temporarily allows
	// pulls to burst to this number, while still not exceeding registryQps.
	// Only used if registryQPS > 0.
	RegistryBurst int32
	// eventRecordQPS is the maximum event creations per second. If 0, there
	// is no limit enforced.
	EventRecordQPS int32
	// eventBurst is the maximum size of a bursty event records, temporarily
	// allows event records to burst to this number, while still not exceeding
	// event-qps. Only used if eventQps > 0
	EventBurst int32
	// enableDebuggingHandlers enables server endpoints for log collection
	// and local running of containers and commands
	EnableDebuggingHandlers bool
	// minimumGCAge is the minimum age for a finished container before it is
	// garbage collected.
	MinimumGCAge metav1.Duration
	// maxPerPodContainerCount is the maximum number of old instances to
	// retain per container. Each container takes up some disk space.
	MaxPerPodContainerCount int32
	// maxContainerCount is the maximum number of old instances of containers
	// to retain globally. Each container takes up some disk space.
	MaxContainerCount int32
	// cAdvisorPort is the port of the localhost cAdvisor endpoint
	CAdvisorPort int32
	// healthzPort is the port of the localhost healthz endpoint
	HealthzPort int32
	// healthzBindAddress is the IP address for the healthz server to serve
	// on.
	HealthzBindAddress string
	// oomScoreAdj is The oom-score-adj value for kubelet process. Values
	// must be within the range [-1000, 1000].
	OOMScoreAdj int32
	// registerNode enables automatic registration with the apiserver.
	RegisterNode bool
	// clusterDomain is the DNS domain for this cluster. If set, kubelet will
	// configure all containers to search this domain in addition to the
	// host's search domains.
	ClusterDomain string
	// masterServiceNamespace is The namespace from which the kubernetes
	// master services should be injected into pods.
	MasterServiceNamespace string
	// clusterDNS is the IP address for a cluster DNS server.  If set, kubelet
	// will configure all containers to use this for DNS resolution in
	// addition to the host's DNS servers
	ClusterDNS string
	// streamingConnectionIdleTimeout is the maximum time a streaming connection
	// can be idle before the connection is automatically closed.
	StreamingConnectionIdleTimeout metav1.Duration
	// nodeStatusUpdateFrequency is the frequency that kubelet posts node
	// status to master. Note: be cautious when changing the constant, it
	// must work with nodeMonitorGracePeriod in nodecontroller.
	NodeStatusUpdateFrequency metav1.Duration
	// imageMinimumGCAge is the minimum age for an unused image before it is
	// garbage collected.
	ImageMinimumGCAge metav1.Duration
	// imageGCHighThresholdPercent is the percent of disk usage after which
	// image garbage collection is always run.
	ImageGCHighThresholdPercent int32
	// imageGCLowThresholdPercent is the percent of disk usage before which
	// image garbage collection is never run. Lowest disk usage to garbage
	// collect to.
	ImageGCLowThresholdPercent int32
	// lowDiskSpaceThresholdMB is the absolute free disk space, in MB, to
	// maintain. When disk space falls below this threshold, new pods would
	// be rejected.
	LowDiskSpaceThresholdMB int32
	// How frequently to calculate and cache volume disk usage for all pods
	VolumeStatsAggPeriod metav1.Duration
	// networkPluginName is the name of the network plugin to be invoked for
	// various events in kubelet/pod lifecycle
	NetworkPluginName string
	// networkPluginMTU is the MTU to be passed to the network plugin,
	// and overrides the default MTU for cases where it cannot be automatically
	// computed (such as IPSEC).
	NetworkPluginMTU int32
	// networkPluginDir is the full path of the directory in which to search
	// for network plugins (and, for backwards-compat, CNI config files)
	NetworkPluginDir string
	// CNIConfDir is the full path of the directory in which to search for
	// CNI config files
	CNIConfDir string
	// CNIBinDir is the full path of the directory in which to search for
	// CNI plugin binaries
	CNIBinDir string
	// volumePluginDir is the full path of the directory in which to search
	// for additional third party volume plugins
	VolumePluginDir string
	// cloudProvider is the provider for cloud services.
	// +optional
	CloudProvider string
	// cloudConfigFile is the path to the cloud provider configuration file.
	// +optional
	CloudConfigFile string
	// KubeletCgroups is the absolute name of cgroups to isolate the kubelet in.
	// +optional
	KubeletCgroups string
	// Enable QoS based Cgroup hierarchy: top level cgroups for QoS Classes
	// And all Burstable and BestEffort pods are brought up under their
	// specific top level QoS cgroup.
	// +optional
	ExperimentalCgroupsPerQOS bool
	// driver that the kubelet uses to manipulate cgroups on the host (cgroupfs or systemd)
	// +optional
	CgroupDriver string
	// Cgroups that container runtime is expected to be isolated in.
	// +optional
	RuntimeCgroups string
	// SystemCgroups is absolute name of cgroups in which to place
	// all non-kernel processes that are not already in a container. Empty
	// for no container. Rolling back the flag requires a reboot.
	// +optional
	SystemCgroups string
	// CgroupRoot is the root cgroup to use for pods.
	// If ExperimentalCgroupsPerQOS is enabled, this is the root of the QoS cgroup hierarchy.
	// +optional
	CgroupRoot string
	// containerRuntime is the container runtime to use.
	ContainerRuntime string
	// remoteRuntimeEndpoint is the endpoint of remote runtime service
	RemoteRuntimeEndpoint string
	// remoteImageEndpoint is the endpoint of remote image service
	RemoteImageEndpoint string
	// runtimeRequestTimeout is the timeout for all runtime requests except long running
	// requests - pull, logs, exec and attach.
	// +optional
	RuntimeRequestTimeout metav1.Duration
	// If no pulling progress is made before the deadline imagePullProgressDeadline,
	// the image pulling will be cancelled. Defaults to 1m0s.
	// +optional
	ImagePullProgressDeadline metav1.Duration
	// rktPath is the path of rkt binary. Leave empty to use the first rkt in
	// $PATH.
	// +optional
	RktPath string
	// experimentalMounterPath is the path of mounter binary. Leave empty to use the default mount path
	ExperimentalMounterPath string
	// rktApiEndpoint is the endpoint of the rkt API service to communicate with.
	// +optional
	RktAPIEndpoint string
	// rktStage1Image is the image to use as stage1. Local paths and
	// http/https URLs are supported.
	// +optional
	RktStage1Image string
	// lockFilePath is the path that kubelet will use to as a lock file.
	// It uses this file as a lock to synchronize with other kubelet processes
	// that may be running.
	LockFilePath string
	// ExitOnLockContention is a flag that signifies to the kubelet that it is running
	// in "bootstrap" mode. This requires that 'LockFilePath' has been set.
	// This will cause the kubelet to listen to inotify events on the lock file,
	// releasing it and exiting when another process tries to open that file.
	ExitOnLockContention bool
	// How should the kubelet configure the container bridge for hairpin packets.
	// Setting this flag allows endpoints in a Service to loadbalance back to
	// themselves if they should try to access their own Service. Values:
	//   "promiscuous-bridge": make the container bridge promiscuous.
	//   "hairpin-veth":       set the hairpin flag on container veth interfaces.
	//   "none":               do nothing.
	// Generally, one must set --hairpin-mode=veth-flag to achieve hairpin NAT,
	// because promiscous-bridge assumes the existence of a container bridge named cbr0.
	HairpinMode string
	// The node has babysitter process monitoring docker and kubelet.
	BabysitDaemons bool
	// maxPods is the number of pods that can run on this Kubelet.
	MaxPods int32
	// nvidiaGPUs is the number of NVIDIA GPU devices on this node.
	NvidiaGPUs int32
	// dockerExecHandlerName is the handler to use when executing a command
	// in a container. Valid values are 'native' and 'nsenter'. Defaults to
	// 'native'.
	DockerExecHandlerName string
	// The CIDR to use for pod IP addresses, only used in standalone mode.
	// In cluster mode, this is obtained from the master.
	PodCIDR string
	// ResolverConfig is the resolver configuration file used as the basis
	// for the container DNS resolution configuration."), []
	ResolverConfig string
	// cpuCFSQuota is Enable CPU CFS quota enforcement for containers that
	// specify CPU limits
	CPUCFSQuota bool
	// containerized should be set to true if kubelet is running in a container.
	Containerized bool
	// maxOpenFiles is Number of files that can be opened by Kubelet process.
	MaxOpenFiles int64
	// registerSchedulable tells the kubelet to register the node as
	// schedulable. Won't have any effect if register-node is false.
	// DEPRECATED: use registerWithTaints instead
	RegisterSchedulable bool
	// registerWithTaints are an array of taints to add to a node object when
	// the kubelet registers itself. This only takes effect when registerNode
	// is true and upon the initial registration of the node.
	RegisterWithTaints []api.Taint
	// contentType is contentType of requests sent to apiserver.
	ContentType string
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver
	KubeAPIQPS int32
	// kubeAPIBurst is the burst to allow while talking with kubernetes
	// apiserver
	KubeAPIBurst int32
	// serializeImagePulls when enabled, tells the Kubelet to pull images one
	// at a time. We recommend *not* changing the default value on nodes that
	// run docker daemon with version  < 1.9 or an Aufs storage backend.
	// Issue #10959 has more details.
	SerializeImagePulls bool
	// outOfDiskTransitionFrequency is duration for which the kubelet has to
	// wait before transitioning out of out-of-disk node condition status.
	// +optional
	OutOfDiskTransitionFrequency metav1.Duration
	// nodeIP is IP address of the node. If set, kubelet will use this IP
	// address for the node.
	// +optional
	NodeIP string
	// nodeLabels to add when registering the node in the cluster.
	NodeLabels map[string]string
	// nonMasqueradeCIDR configures masquerading: traffic to IPs outside this range will use IP masquerade.
	NonMasqueradeCIDR string
	// enable gathering custom metrics.
	EnableCustomMetrics bool
	// Comma-delimited list of hard eviction expressions.  For example, 'memory.available<300Mi'.
	// +optional
	EvictionHard string
	// Comma-delimited list of soft eviction expressions.  For example, 'memory.available<300Mi'.
	// +optional
	EvictionSoft string
	// Comma-delimeted list of grace periods for each soft eviction signal.  For example, 'memory.available=30s'.
	// +optional
	EvictionSoftGracePeriod string
	// Duration for which the kubelet has to wait before transitioning out of an eviction pressure condition.
	// +optional
	EvictionPressureTransitionPeriod metav1.Duration
	// Maximum allowed grace period (in seconds) to use when terminating pods in response to a soft eviction threshold being met.
	// +optional
	EvictionMaxPodGracePeriod int32
	// Comma-delimited list of minimum reclaims (e.g. imagefs.available=2Gi) that describes the minimum amount of resource the kubelet will reclaim when performing a pod eviction if that resource is under pressure.
	// +optional
	EvictionMinimumReclaim string
	// If enabled, the kubelet will integrate with the kernel memcg notification to determine if memory eviction thresholds are crossed rather than polling.
	// +optional
	ExperimentalKernelMemcgNotification bool
	// Maximum number of pods per core. Cannot exceed MaxPods
	PodsPerCore int32
	// enableControllerAttachDetach enables the Attach/Detach controller to
	// manage attachment/detachment of volumes scheduled to this node, and
	// disables kubelet from executing any attach/detach operations
	EnableControllerAttachDetach bool
	// A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs
	// that describe resources reserved for non-kubernetes components.
	// Currently only cpu and memory are supported. [default=none]
	// See http://kubernetes.io/docs/user-guide/compute-resources for more detail.
	SystemReserved utilconfig.ConfigurationMap
	// A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs
	// that describe resources reserved for kubernetes system components.
	// Currently only cpu and memory are supported. [default=none]
	// See http://kubernetes.io/docs/user-guide/compute-resources for more detail.
	KubeReserved utilconfig.ConfigurationMap
	// Default behaviour for kernel tuning
	ProtectKernelDefaults bool
	// If true, Kubelet ensures a set of iptables rules are present on host.
	// These rules will serve as utility for various components, e.g. kube-proxy.
	// The rules will be created based on IPTablesMasqueradeBit and IPTablesDropBit.
	MakeIPTablesUtilChains bool
	// iptablesMasqueradeBit is the bit of the iptables fwmark space to use for SNAT
	// Values must be within the range [0, 31].
	// Warning: Please match the value of corresponding parameter in kube-proxy
	// TODO: clean up IPTablesMasqueradeBit in kube-proxy
	IPTablesMasqueradeBit int32
	// iptablesDropBit is the bit of the iptables fwmark space to use for dropping packets. Kubelet will ensure iptables mark and drop rules.
	// Values must be within the range [0, 31]. Must be different from IPTablesMasqueradeBit
	IPTablesDropBit int32
	// Whitelist of unsafe sysctls or sysctl patterns (ending in *).
	// +optional
	AllowedUnsafeSysctls []string
	// featureGates is a string of comma-separated key=value pairs that describe feature
	// gates for alpha/experimental features.
	FeatureGates string
	// Enable Container Runtime Interface (CRI) integration.
	// +optional
	EnableCRI bool
	// TODO(#34726:1.8.0): Remove the opt-in for failing when swap is enabled.
	// Tells the Kubelet to fail to start if swap is enabled on the node.
	ExperimentalFailSwapOn bool
	// This flag, if set, enables a check prior to mount operations to verify that the required components
	// (binaries, etc.) to mount the volume are available on the underlying node. If the check is enabled
	// and fails the mount operation fails.
	ExperimentalCheckNodeCapabilitiesBeforeMount bool
}

type KubeletAuthorizationMode string

const (
	// KubeletAuthorizationModeAlwaysAllow authorizes all authenticated requests
	KubeletAuthorizationModeAlwaysAllow KubeletAuthorizationMode = "AlwaysAllow"
	// KubeletAuthorizationModeWebhook uses the SubjectAccessReview API to determine authorization
	KubeletAuthorizationModeWebhook KubeletAuthorizationMode = "Webhook"
)

type KubeletAuthorization struct {
	// mode is the authorization mode to apply to requests to the kubelet server.
	// Valid values are AlwaysAllow and Webhook.
	// Webhook mode uses the SubjectAccessReview API to determine authorization.
	Mode KubeletAuthorizationMode

	// webhook contains settings related to Webhook authorization.
	Webhook KubeletWebhookAuthorization
}

type KubeletWebhookAuthorization struct {
	// cacheAuthorizedTTL is the duration to cache 'authorized' responses from the webhook authorizer.
	CacheAuthorizedTTL metav1.Duration
	// cacheUnauthorizedTTL is the duration to cache 'unauthorized' responses from the webhook authorizer.
	CacheUnauthorizedTTL metav1.Duration
}

type KubeletAuthentication struct {
	// x509 contains settings related to x509 client certificate authentication
	X509 KubeletX509Authentication
	// webhook contains settings related to webhook bearer token authentication
	Webhook KubeletWebhookAuthentication
	// anonymous contains settings related to anonymous authentication
	Anonymous KubeletAnonymousAuthentication
}

type KubeletX509Authentication struct {
	// clientCAFile is the path to a PEM-encoded certificate bundle. If set, any request presenting a client certificate
	// signed by one of the authorities in the bundle is authenticated with a username corresponding to the CommonName,
	// and groups corresponding to the Organization in the client certificate.
	ClientCAFile string
}

type KubeletWebhookAuthentication struct {
	// enabled allows bearer token authentication backed by the tokenreviews.authentication.k8s.io API
	Enabled bool
	// cacheTTL enables caching of authentication results
	CacheTTL metav1.Duration
}

type KubeletAnonymousAuthentication struct {
	// enabled allows anonymous requests to the kubelet server.
	// Requests that are not rejected by another authentication method are treated as anonymous requests.
	// Anonymous requests have a username of system:anonymous, and a group name of system:unauthenticated.
	Enabled bool
}

type KubeSchedulerConfiguration struct {
	metav1.TypeMeta

	// port is the port that the scheduler's http service runs on.
	Port int32
	// address is the IP address to serve on.
	Address string
	// algorithmProvider is the scheduling algorithm provider to use.
	AlgorithmProvider string
	// policyConfigFile is the filepath to the scheduler policy configuration.
	PolicyConfigFile string
	// enableProfiling enables profiling via web interface.
	EnableProfiling bool
	// enableContentionProfiling enables lock contention profiling, if enableProfiling is true.
	EnableContentionProfiling bool
	// contentType is contentType of requests sent to apiserver.
	ContentType string
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver.
	KubeAPIQPS float32
	// kubeAPIBurst is the QPS burst to use while talking with kubernetes apiserver.
	KubeAPIBurst int32
	// schedulerName is name of the scheduler, used to select which pods
	// will be processed by this scheduler, based on pod's annotation with
	// key 'scheduler.alpha.kubernetes.io/name'.
	SchedulerName string
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	HardPodAffinitySymmetricWeight int
	// Indicate the "all topologies" set for empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.
	FailureDomains string
	// leaderElection defines the configuration of leader election client.
	LeaderElection LeaderElectionConfiguration
}

// LeaderElectionConfiguration defines the configuration of leader election
// clients for components that can run with leader election enabled.
type LeaderElectionConfiguration struct {
	// leaderElect enables a leader election client to gain leadership
	// before executing the main loop. Enable this when running replicated
	// components for high availability.
	LeaderElect bool
	// leaseDuration is the duration that non-leader candidates will wait
	// after observing a leadership renewal until attempting to acquire
	// leadership of a led but unrenewed leader slot. This is effectively the
	// maximum duration that a leader can be stopped before it is replaced
	// by another candidate. This is only applicable if leader election is
	// enabled.
	LeaseDuration metav1.Duration
	// renewDeadline is the interval between attempts by the acting master to
	// renew a leadership slot before it stops leading. This must be less
	// than or equal to the lease duration. This is only applicable if leader
	// election is enabled.
	RenewDeadline metav1.Duration
	// retryPeriod is the duration the clients should wait between attempting
	// acquisition and renewal of a leadership. This is only applicable if
	// leader election is enabled.
	RetryPeriod metav1.Duration
}

type KubeControllerManagerConfiguration struct {
	metav1.TypeMeta

	// Controllers is the list of controllers to enable or disable
	// '*' means "all enabled by default controllers"
	// 'foo' means "enable 'foo'"
	// '-foo' means "disable 'foo'"
	// first item for a particular name wins
	Controllers []string

	// port is the port that the controller-manager's http service runs on.
	Port int32
	// address is the IP address to serve on (set to 0.0.0.0 for all interfaces).
	Address string
	// useServiceAccountCredentials indicates whether controllers should be run with
	// individual service account credentials.
	UseServiceAccountCredentials bool
	// cloudProvider is the provider for cloud services.
	CloudProvider string
	// cloudConfigFile is the path to the cloud provider configuration file.
	CloudConfigFile string
	// concurrentEndpointSyncs is the number of endpoint syncing operations
	// that will be done concurrently. Larger number = faster endpoint updating,
	// but more CPU (and network) load.
	ConcurrentEndpointSyncs int32
	// concurrentRSSyncs is the number of replica sets that are  allowed to sync
	// concurrently. Larger number = more responsive replica  management, but more
	// CPU (and network) load.
	ConcurrentRSSyncs int32
	// concurrentRCSyncs is the number of replication controllers that are
	// allowed to sync concurrently. Larger number = more responsive replica
	// management, but more CPU (and network) load.
	ConcurrentRCSyncs int32
	// concurrentServiceSyncs is the number of services that are
	// allowed to sync concurrently. Larger number = more responsive service
	// management, but more CPU (and network) load.
	ConcurrentServiceSyncs int32
	// concurrentResourceQuotaSyncs is the number of resource quotas that are
	// allowed to sync concurrently. Larger number = more responsive quota
	// management, but more CPU (and network) load.
	ConcurrentResourceQuotaSyncs int32
	// concurrentDeploymentSyncs is the number of deployment objects that are
	// allowed to sync concurrently. Larger number = more responsive deployments,
	// but more CPU (and network) load.
	ConcurrentDeploymentSyncs int32
	// concurrentDaemonSetSyncs is the number of daemonset objects that are
	// allowed to sync concurrently. Larger number = more responsive daemonset,
	// but more CPU (and network) load.
	ConcurrentDaemonSetSyncs int32
	// concurrentJobSyncs is the number of job objects that are
	// allowed to sync concurrently. Larger number = more responsive jobs,
	// but more CPU (and network) load.
	ConcurrentJobSyncs int32
	// concurrentNamespaceSyncs is the number of namespace objects that are
	// allowed to sync concurrently.
	ConcurrentNamespaceSyncs int32
	// concurrentSATokenSyncs is the number of service account token syncing operations
	// that will be done concurrently.
	ConcurrentSATokenSyncs int32
	// lookupCacheSizeForRC is the size of lookup cache for replication controllers.
	// Larger number = more responsive replica management, but more MEM load.
	LookupCacheSizeForRC int32
	// lookupCacheSizeForRS is the size of lookup cache for replicatsets.
	// Larger number = more responsive replica management, but more MEM load.
	LookupCacheSizeForRS int32
	// lookupCacheSizeForDaemonSet is the size of lookup cache for daemonsets.
	// Larger number = more responsive daemonset, but more MEM load.
	LookupCacheSizeForDaemonSet int32
	// serviceSyncPeriod is the period for syncing services with their external
	// load balancers.
	ServiceSyncPeriod metav1.Duration
	// nodeSyncPeriod is the period for syncing nodes from cloudprovider. Longer
	// periods will result in fewer calls to cloud provider, but may delay addition
	// of new nodes to cluster.
	NodeSyncPeriod metav1.Duration
	// routeReconciliationPeriod is the period for reconciling routes created for Nodes by cloud provider..
	RouteReconciliationPeriod metav1.Duration
	// resourceQuotaSyncPeriod is the period for syncing quota usage status
	// in the system.
	ResourceQuotaSyncPeriod metav1.Duration
	// namespaceSyncPeriod is the period for syncing namespace life-cycle
	// updates.
	NamespaceSyncPeriod metav1.Duration
	// pvClaimBinderSyncPeriod is the period for syncing persistent volumes
	// and persistent volume claims.
	PVClaimBinderSyncPeriod metav1.Duration
	// minResyncPeriod is the resync period in reflectors; will be random between
	// minResyncPeriod and 2*minResyncPeriod.
	MinResyncPeriod metav1.Duration
	// terminatedPodGCThreshold is the number of terminated pods that can exist
	// before the terminated pod garbage collector starts deleting terminated pods.
	// If <= 0, the terminated pod garbage collector is disabled.
	TerminatedPodGCThreshold int32
	// horizontalPodAutoscalerSyncPeriod is the period for syncing the number of
	// pods in horizontal pod autoscaler.
	HorizontalPodAutoscalerSyncPeriod metav1.Duration
	// deploymentControllerSyncPeriod is the period for syncing the deployments.
	DeploymentControllerSyncPeriod metav1.Duration
	// podEvictionTimeout is the grace period for deleting pods on failed nodes.
	PodEvictionTimeout metav1.Duration
	// DEPRECATED: deletingPodsQps is the number of nodes per second on which pods are deleted in
	// case of node failure.
	DeletingPodsQps float32
	// DEPRECATED: deletingPodsBurst is the number of nodes on which pods are bursty deleted in
	// case of node failure. For more details look into RateLimiter.
	DeletingPodsBurst int32
	// nodeMontiorGracePeriod is the amount of time which we allow a running node to be
	// unresponsive before marking it unhealthy. Must be N times more than kubelet's
	// nodeStatusUpdateFrequency, where N means number of retries allowed for kubelet
	// to post node status.
	NodeMonitorGracePeriod metav1.Duration
	// registerRetryCount is the number of retries for initial node registration.
	// Retry interval equals node-sync-period.
	RegisterRetryCount int32
	// nodeStartupGracePeriod is the amount of time which we allow starting a node to
	// be unresponsive before marking it unhealthy.
	NodeStartupGracePeriod metav1.Duration
	// nodeMonitorPeriod is the period for syncing NodeStatus in NodeController.
	NodeMonitorPeriod metav1.Duration
	// serviceAccountKeyFile is the filename containing a PEM-encoded private RSA key
	// used to sign service account tokens.
	ServiceAccountKeyFile string
	// clusterSigningCertFile is the filename containing a PEM-encoded
	// X509 CA certificate used to issue cluster-scoped certificates
	ClusterSigningCertFile string
	// clusterSigningCertFile is the filename containing a PEM-encoded
	// RSA or ECDSA private key used to issue cluster-scoped certificates
	ClusterSigningKeyFile string
	// approveAllKubeletCSRs tells the CSR controller to approve all CSRs originating
	// from the kubelet bootstrapping group automatically.
	// WARNING: this grants all users with access to the certificates API group
	// the ability to create credentials for any user that has access to the boostrapping
	// user's credentials.
	ApproveAllKubeletCSRsForGroup string
	// enableProfiling enables profiling via web interface host:port/debug/pprof/
	EnableProfiling bool
	// clusterName is the instance prefix for the cluster.
	ClusterName string
	// clusterCIDR is CIDR Range for Pods in cluster.
	ClusterCIDR string
	// serviceCIDR is CIDR Range for Services in cluster.
	ServiceCIDR string
	// NodeCIDRMaskSize is the mask size for node cidr in cluster.
	NodeCIDRMaskSize int32
	// allocateNodeCIDRs enables CIDRs for Pods to be allocated and, if
	// ConfigureCloudRoutes is true, to be set on the cloud provider.
	AllocateNodeCIDRs bool
	// configureCloudRoutes enables CIDRs allocated with allocateNodeCIDRs
	// to be configured on the cloud provider.
	ConfigureCloudRoutes bool
	// rootCAFile is the root certificate authority will be included in service
	// account's token secret. This must be a valid PEM-encoded CA bundle.
	RootCAFile string
	// contentType is contentType of requests sent to apiserver.
	ContentType string
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver.
	KubeAPIQPS float32
	// kubeAPIBurst is the burst to use while talking with kubernetes apiserver.
	KubeAPIBurst int32
	// leaderElection defines the configuration of leader election client.
	LeaderElection LeaderElectionConfiguration
	// volumeConfiguration holds configuration for volume related features.
	VolumeConfiguration VolumeConfiguration
	// How long to wait between starting controller managers
	ControllerStartInterval metav1.Duration
	// enables the generic garbage collector. MUST be synced with the
	// corresponding flag of the kube-apiserver. WARNING: the generic garbage
	// collector is an alpha feature.
	EnableGarbageCollector bool
	// concurrentGCSyncs is the number of garbage collector workers that are
	// allowed to sync concurrently.
	ConcurrentGCSyncs int32
	// nodeEvictionRate is the number of nodes per second on which pods are deleted in case of node failure when a zone is healthy
	NodeEvictionRate float32
	// secondaryNodeEvictionRate is the number of nodes per second on which pods are deleted in case of node failure when a zone is unhealty
	SecondaryNodeEvictionRate float32
	// secondaryNodeEvictionRate is implicitly overridden to 0 for clusters smaller than or equal to largeClusterSizeThreshold
	LargeClusterSizeThreshold int32
	// Zone is treated as unhealthy in nodeEvictionRate and secondaryNodeEvictionRate when at least
	// unhealthyZoneThreshold (no less than 3) of Nodes in the zone are NotReady
	UnhealthyZoneThreshold float32
	// Reconciler runs a periodic loop to reconcile the desired state of the with
	// the actual state of the world by triggering attach detach operations.
	// This flag enables or disables reconcile.  Is false by default, and thus enabled.
	DisableAttachDetachReconcilerSync bool
	// ReconcilerSyncLoopPeriod is the amount of time the reconciler sync states loop
	// wait between successive executions. Is set to 5 sec by default.
	ReconcilerSyncLoopPeriod metav1.Duration
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
	EnableHostPathProvisioning bool
	// enableDynamicProvisioning enables the provisioning of volumes when running within an environment
	// that supports dynamic provisioning. Defaults to true.
	EnableDynamicProvisioning bool
	// persistentVolumeRecyclerConfiguration holds configuration for persistent volume plugins.
	PersistentVolumeRecyclerConfiguration PersistentVolumeRecyclerConfiguration
	// volumePluginDir is the full path of the directory in which the flex
	// volume plugin should search for additional third party volume plugins
	FlexVolumePluginDir string
}

type PersistentVolumeRecyclerConfiguration struct {
	// maximumRetry is number of retries the PV recycler will execute on failure to recycle
	// PV.
	MaximumRetry int32
	// minimumTimeoutNFS is the minimum ActiveDeadlineSeconds to use for an NFS Recycler
	// pod.
	MinimumTimeoutNFS int32
	// podTemplateFilePathNFS is the file path to a pod definition used as a template for
	// NFS persistent volume recycling
	PodTemplateFilePathNFS string
	// incrementTimeoutNFS is the increment of time added per Gi to ActiveDeadlineSeconds
	// for an NFS scrubber pod.
	IncrementTimeoutNFS int32
	// podTemplateFilePathHostPath is the file path to a pod definition used as a template for
	// HostPath persistent volume recycling. This is for development and testing only and
	// will not work in a multi-node cluster.
	PodTemplateFilePathHostPath string
	// minimumTimeoutHostPath is the minimum ActiveDeadlineSeconds to use for a HostPath
	// Recycler pod.  This is for development and testing only and will not work in a multi-node
	// cluster.
	MinimumTimeoutHostPath int32
	// incrementTimeoutHostPath is the increment of time added per Gi to ActiveDeadlineSeconds
	// for a HostPath scrubber pod.  This is for development and testing only and will not work
	// in a multi-node cluster.
	IncrementTimeoutHostPath int32
}

// AdmissionConfiguration provides versioned configuration for admission controllers.
type AdmissionConfiguration struct {
	metav1.TypeMeta

	// Plugins allows specifying a configuration per admission control plugin.
	Plugins []AdmissionPluginConfiguration
}

// AdmissionPluginConfiguration provides the configuration for a single plug-in.
type AdmissionPluginConfiguration struct {
	// Name is the name of the admission controller.
	// It must match the registered admission plugin name.
	Name string

	// Path is the path to a configuration file that contains the plugin's
	// configuration
	// +optional
	Path string

	// Configuration is an embedded configuration object to be used as the plugin's
	// configuration. If present, it will be used instead of the path to the configuration file.
	// +optional
	Configuration runtime.Object
}
