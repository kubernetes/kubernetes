/*
Copyright 2017 The Kubernetes Authors.

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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
)

// A configuration field should go in KubeletFlags instead of KubeletConfiguration if any of these are true:
// - its value will never, or cannot safely be changed during the lifetime of a node
// - its value cannot be safely shared between nodes at the same time (e.g. a hostname)
//   KubeletConfiguration is intended to be shared between nodes
// In general, please try to avoid adding flags or configuration fields,
// we already have a confusingly large amount of them.
type KubeletConfiguration struct {
	metav1.TypeMeta // TODO(mtaufen): remove TypeMeta from this aggregate
	NodeLifecycleConfig
	PodLifecycleConfig
	NodeMonitoringConfig
	KubeletServerConfig
	NodeResourcesConfig
	NodeSecurityConfig
	NodeNetworkConfig
	ClusterFeatureGates
	ClusterAPIConfig
	ImageAPIConfig
	KubeletDebugConfig
}

// TODO(mtaufen): The registration stuff probably belongs only in an init config, since they only matter when the node is first registered
type NodeLifecycleConfig struct {
	metav1.TypeMeta `json:",inline"`
	// Default behaviour for kernel tuning
	ProtectKernelDefaults bool `json:"protectKernelDefaults"` // If true, Kubelet will error if kernel parameters aren't what it expects. If false, Kubelet will try to set kernel parameters to what it wants.
	// registerNode enables automatic registration with the apiserver.
	RegisterNode *bool `json:"registerNode"`
	// registerWithTaints are an array of taints to add to a node object when
	// the kubelet registers itself. This only takes effect when registerNode
	// is true and upon the initial registration of the node.
	RegisterWithTaints []v1.Taint `json:"registerWithTaints"`
	// nodeLabels to add when registering the node in the cluster.
	NodeLabels map[string]string `json:"nodeLabels"`
}

type PodLifecycleConfig struct {
	metav1.TypeMeta `json:",inline"`
	// syncFrequency is the max period between synchronizing running
	// containers and config
	SyncFrequency metav1.Duration `json:"syncFrequency"`
}

type NodeMonitoringConfig struct {
	metav1.TypeMeta `json:",inline"`
	// nodeStatusUpdateFrequency is the frequency that kubelet posts node
	// status to master. Note: be cautious when changing the constant, it
	// must work with nodeMonitorGracePeriod in nodecontroller.
	NodeStatusUpdateFrequency metav1.Duration `json:"nodeStatusUpdateFrequency"`
	// eventRecordQPS is the maximum event creations per second. If 0, there
	// is no limit enforced.
	EventRecordQPS *int32 `json:"eventRecordQPS"`
	// eventBurst is the maximum size of a bursty event records, temporarily
	// allows event records to burst to this number, while still not exceeding
	// event-qps. Only used if eventQps > 0
	EventBurst int32 `json:"eventBurst"`
	// How frequently to calculate and cache volume disk usage for all pods
	VolumeStatsAggPeriod metav1.Duration `json:"volumeStatsAggPeriod"`
	// enable gathering custom metrics.
	EnableCustomMetrics bool `json:"enableCustomMetrics"`
}
type KubeletServerConfig struct {
	metav1.TypeMeta `json:",inline"`
	// enableServer enables the Kubelet's server
	EnableServer *bool `json:"enableServer"`
	// address is the IP address for the Kubelet to serve on (set to 0.0.0.0
	// for all interfaces)
	Address string `json:"address"`
	// port is the port for the Kubelet to serve on.
	Port int32 `json:"port"`
	// readOnlyPort is the read-only port for the Kubelet to serve on with
	// no authentication/authorization (set to 0 to disable)
	ReadOnlyPort int32 `json:"readOnlyPort"`
	// TODO(mtaufen): This should be renamed, because it's actually turned on by default
	// and is necessary for kubectl features to work. We should split out all things
	// that are really "debug mode only" and call this something like `EnableContainerHandlers`.
	// enableDebuggingHandlers enables server endpoints for log collection
	// and local running of containers and commands
	EnableDebuggingHandlers *bool `json:"enableDebuggingHandlers"`
	// streamingConnectionIdleTimeout is the maximum time a streaming connection
	// can be idle before the connection is automatically closed.
	StreamingConnectionIdleTimeout metav1.Duration `json:"streamingConnectionIdleTimeout"`
	// cAdvisorPort is the port of the localhost cAdvisor endpoint
	CAdvisorPort int32 `json:"cAdvisorPort"`
	// healthzPort is the port of the localhost healthz endpoint
	HealthzPort int32 `json:"healthzPort"`
	// healthzBindAddress is the IP address for the healthz server to serve
	// on.
	HealthzBindAddress string `json:"healthzBindAddress"`

	/* auth */
	// authentication specifies how requests to the Kubelet's server are authenticated
	Authentication KubeletAuthentication `json:"authentication"`
	// authorization specifies how requests to the Kubelet's server are authorized
	Authorization KubeletAuthorization `json:"authorization"`

	/* TLS */
	// tlsCertFile is the file containing x509 Certificate for HTTPS.  (CA cert,
	// if any, concatenated after server cert). If tlsCertFile and
	// tlsPrivateKeyFile are not provided, a self-signed certificate
	// and key are generated for the public address and saved to the directory
	// passed to certDir.
	TLSCertFile string `json:"tlsCertFile"`
	// tlsPrivateKeyFile is the ile containing x509 private key matching
	// tlsCertFile.
	TLSPrivateKeyFile string `json:"tlsPrivateKeyFile"`
	// certDirectory is the directory where the TLS certs are located (by
	// default /var/run/kubernetes). If tlsCertFile and tlsPrivateKeyFile
	// are provided, this flag will be ignored.
	CertDirectory string `json:"certDirectory"`
}

type NodeResourcesConfig struct {
	metav1.TypeMeta `json:",inline"`

	/* general resource control */
	// oomScoreAdj is The oom-score-adj value for kubelet process. Values
	// must be within the range [-1000, 1000].
	OOMScoreAdj *int32 `json:"oomScoreAdj"`
	// maxPods is the number of pods that can run on this Kubelet.
	MaxPods int32 `json:"maxPods"`
	// Maximum number of pods per core. Cannot exceed MaxPods
	PodsPerCore int32 `json:"podsPerCore"`
	// cpuCFSQuota is Enable CPU CFS quota enforcement for containers that
	// specify CPU limits
	CPUCFSQuota *bool `json:"cpuCFSQuota"`
	// maxOpenFiles is Number of files that can be opened by Kubelet process.
	MaxOpenFiles int64 `json:"maxOpenFiles"`

	/* cgroups */
	// TODO(mtaufen): I put the cgroups stuff in flags for now, since I'm not sure if it's safe to dynamically change it (including cgroup stuff for node allocatable)

	/* node allocatable */
	// A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs
	// that describe resources reserved for non-kubernetes components.
	// Currently only cpu and memory are supported. [default=none]
	// See http://kubernetes.io/docs/user-guide/compute-resources for more detail.
	SystemReserved map[string]string `json:"systemReserved"`
	// A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs
	// that describe resources reserved for kubernetes system components.
	// Currently only cpu and memory are supported. [default=none]
	// See http://kubernetes.io/docs/user-guide/compute-resources for more detail.
	KubeReserved map[string]string `json:"kubeReserved"`
	// This flag specifies the various Node Allocatable enforcements that Kubelet needs to perform.
	// This flag accepts a list of options. Acceptible options are `pods`, `system-reserved` & `kube-reserved`.
	// Refer to [Node Allocatable](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/node-allocatable.md) doc for more information.
	EnforceNodeAllocatable []string `json:"enforceNodeAllocatable,omitempty"`
	// This flag, if set, will avoid including `EvictionHard` limits while computing Node Allocatable.
	// Refer to [Node Allocatable](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/node-allocatable.md) doc for more information.
	ExperimentalNodeAllocatableIgnoreEvictionThreshold bool `json:"experimentalNodeAllocatableIgnoreEvictionThreshold,omitempty"`

	/* container image garbage collection */
	// imageMinimumGCAge is the minimum age for an unused image before it is
	// garbage collected.
	ImageMinimumGCAge metav1.Duration `json:"imageMinimumGCAge"`
	// imageGCHighThresholdPercent is the percent of disk usage after which
	// image garbage collection is always run. The percent is calculated as
	// this field value out of 100.
	ImageGCHighThresholdPercent *int32 `json:"imageGCHighThresholdPercent"`
	// imageGCLowThresholdPercent is the percent of disk usage before which
	// image garbage collection is never run. Lowest disk usage to garbage
	// collect to. The percent is calculated as this field value out of 100.
	ImageGCLowThresholdPercent *int32 `json:"imageGCLowThresholdPercent"`

	/* pod eviction */
	// Comma-delimited list of hard eviction expressions.  For example, 'memory.available<300Mi'.
	EvictionHard *string `json:"evictionHard"`
	// Comma-delimited list of soft eviction expressions.  For example, 'memory.available<300Mi'.
	EvictionSoft string `json:"evictionSoft"`
	// Comma-delimeted list of grace periods for each soft eviction signal.  For example, 'memory.available=30s'.
	EvictionSoftGracePeriod string `json:"evictionSoftGracePeriod"`
	// Duration for which the kubelet has to wait before transitioning out of an eviction pressure condition.
	EvictionPressureTransitionPeriod metav1.Duration `json:"evictionPressureTransitionPeriod"`
	// Maximum allowed grace period (in seconds) to use when terminating pods in response to a soft eviction threshold being met.
	EvictionMaxPodGracePeriod int32 `json:"evictionMaxPodGracePeriod"`
	// Comma-delimited list of minimum reclaims (e.g. imagefs.available=2Gi) that describes the minimum amount of resource the kubelet will reclaim when performing a pod eviction if that resource is under pressure.
	EvictionMinimumReclaim string `json:"evictionMinimumReclaim"`
}
type NodeSecurityConfig struct {
	metav1.TypeMeta `json:",inline"`
	// allowPrivileged enables containers to request privileged mode.
	// Defaults to false.
	AllowPrivileged *bool `json:"allowPrivileged"`
	// seccompProfileRoot is the directory path for seccomp profiles.
	SeccompProfileRoot string `json:"seccompProfileRoot"`
	// Whitelist of unsafe sysctls or sysctl patterns (ending in *). Use these at your own risk.
	// Resource isolation might be lacking and pod might influence each other on the same node.
	// +optional
	AllowedUnsafeSysctls []string `json:"allowedUnsafeSysctls,omitempty"`
	// hostNetworkSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use of host network. Defaults to "*". Valid
	// options are "file", "http", "api", and "*" (all sources).
	HostNetworkSources []string `json:"hostNetworkSources"`
	// hostPIDSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use the host pid namespace. Defaults to "*".
	HostPIDSources []string `json:"hostPIDSources"`
	// hostIPCSources is a comma-separated list of sources from which the
	// Kubelet allows pods to use the host ipc namespace. Defaults to "*".
	HostIPCSources []string `json:"hostIPCSources"`
}
type NodeNetworkConfig struct {
	metav1.TypeMeta `json:",inline"`
	// clusterDNS is a list of IP address for the cluster DNS server.  If set,
	// kubelet will configure all containers to use this for DNS resolution
	// instead of the host's DNS servers
	ClusterDNS []string `json:"clusterDNS"`
	// clusterDomain is the DNS domain for this cluster. If set, kubelet will
	// configure all containers to search this domain in addition to the
	// host's search domains.
	ClusterDomain string `json:"clusterDomain"`
	// ResolverConfig is the resolver configuration file used as the basis
	// for the container DNS resolution configuration."), []
	ResolverConfig string `json:"resolvConf"`
	// nonMasqueradeCIDR configures masquerading: traffic to IPs outside this range will use IP masquerade.
	NonMasqueradeCIDR string `json:"nonMasqueradeCIDR"`
	// If true, Kubelet ensures a set of iptables rules are present on host.
	// These rules will serve as utility rules for various components, e.g. KubeProxy.
	// The rules will be created based on IPTablesMasqueradeBit and IPTablesDropBit.
	MakeIPTablesUtilChains *bool `json:"makeIPTablesUtilChains"`
	// iptablesMasqueradeBit is the bit of the iptables fwmark space to mark for SNAT
	// Values must be within the range [0, 31]. Must be different from other mark bits.
	// Warning: Please match the value of corresponding parameter in kube-proxy
	// TODO: clean up IPTablesMasqueradeBit in kube-proxy
	IPTablesMasqueradeBit *int32 `json:"iptablesMasqueradeBit"`
	// iptablesDropBit is the bit of the iptables fwmark space to mark for dropping packets.
	// Values must be within the range [0, 31]. Must be different from other mark bits.
	IPTablesDropBit *int32 `json:"iptablesDropBit"`
	// How should the kubelet configure the container bridge for hairpin packets.
	// Setting this flag allows endpoints in a Service to loadbalance back to
	// themselves if they should try to access their own Service. Values:
	//   "promiscuous-bridge": make the container bridge promiscuous.
	//   "hairpin-veth":       set the hairpin flag on container veth interfaces.
	//   "none":               do nothing.
	// Generally, one must set --hairpin-mode=veth-flag to achieve hairpin NAT,
	// because promiscous-bridge assumes the existence of a container bridge named cbr0.
	HairpinMode string `json:"hairpinMode"`
	// The CIDR to use for pod IP addresses, only used in standalone mode.
	// In cluster mode, this is obtained from the master.
	PodCIDR string `json:"podCIDR"`
}

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

// TODO(kc-refactor): This will probably be our best opportunity to stop parsing a string
// and make the feature gates API better. We should think about how to do that.
type ClusterFeatureGates struct {
	metav1.TypeMeta `json:",inline"`
	// featureGates is a string of comma-separated key=value pairs that describe feature
	// gates for alpha/experimental features.
	FeatureGates string `json:"featureGates,omitempty"`
}
type ClusterAPIConfig struct {
	metav1.TypeMeta `json:",inline"`
	// contentType is contentType of requests sent to apiserver.
	ContentType string `json:"contentType"`
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver
	KubeAPIQPS *int32 `json:"kubeAPIQPS"`
	// kubeAPIBurst is the burst to allow while talking with kubernetes
	// apiserver
	KubeAPIBurst int32 `json:"kubeAPIBurst"`
}
type ImageAPIConfig struct {
	metav1.TypeMeta `json:",inline"`
	// serializeImagePulls when enabled, tells the Kubelet to pull images one
	// at a time. We recommend *not* changing the default value on nodes that
	// run docker daemon with version  < 1.9 or an Aufs storage backend.
	// Issue #10959 has more details.
	SerializeImagePulls *bool `json:"serializeImagePulls"`
	// registryPullQPS is the limit of registry pulls per second. If 0,
	// unlimited. Set to 0 for no limit. Defaults to 5.0.
	RegistryPullQPS *int32 `json:"registryPullQPS"`
	// registryBurst is the maximum size of a bursty pulls, temporarily allows
	// pulls to burst to this number, while still not exceeding registryQps.
	// Only used if registryQPS > 0.
	RegistryBurst int32 `json:"registryBurst"`
	// If no pulling progress is made before the deadline imagePullProgressDeadline,
	// the image pulling will be cancelled. Defaults to 1m0s.
	ImagePullProgressDeadline metav1.Duration `json:"imagePullProgressDeadline,omitempty"`
}

type KubeletDebugConfig struct {
	metav1.TypeMeta `json:",inline"`
	// enableContentionProfiling enables lock contention profiling, if enableDebuggingHandlers is true in the KubeletServerConfig.
	EnableContentionProfiling bool `json:"enableContentionProfiling"`
	// This flag, if set, instructs the kubelet to keep volumes from terminated pods mounted to the node.
	// This can be useful for debugging volume related issues.
	KeepTerminatedPodVolumes bool `json:"keepTerminatedPodVolumes,omitempty"`
}

//////////
// auth //
//////////
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
	Mode KubeletAuthorizationMode `json:"mode"`

	// webhook contains settings related to Webhook authorization.
	Webhook KubeletWebhookAuthorization `json:"webhook"`
}

type KubeletWebhookAuthorization struct {
	// cacheAuthorizedTTL is the duration to cache 'authorized' responses from the webhook authorizer.
	CacheAuthorizedTTL metav1.Duration `json:"cacheAuthorizedTTL"`
	// cacheUnauthorizedTTL is the duration to cache 'unauthorized' responses from the webhook authorizer.
	CacheUnauthorizedTTL metav1.Duration `json:"cacheUnauthorizedTTL"`
}

type KubeletAuthentication struct {
	// x509 contains settings related to x509 client certificate authentication
	X509 KubeletX509Authentication `json:"x509"`
	// webhook contains settings related to webhook bearer token authentication
	Webhook KubeletWebhookAuthentication `json:"webhook"`
	// anonymous contains settings related to anonymous authentication
	Anonymous KubeletAnonymousAuthentication `json:"anonymous"`
}

type KubeletX509Authentication struct {
	// clientCAFile is the path to a PEM-encoded certificate bundle. If set, any request presenting a client certificate
	// signed by one of the authorities in the bundle is authenticated with a username corresponding to the CommonName,
	// and groups corresponding to the Organization in the client certificate.
	ClientCAFile string `json:"clientCAFile"`
}

type KubeletWebhookAuthentication struct {
	// enabled allows bearer token authentication backed by the tokenreviews.authentication.k8s.io API
	Enabled *bool `json:"enabled"`
	// cacheTTL enables caching of authentication results
	CacheTTL metav1.Duration `json:"cacheTTL"`
}

type KubeletAnonymousAuthentication struct {
	// enabled allows anonymous requests to the kubelet server.
	// Requests that are not rejected by another authentication method are treated as anonymous requests.
	// Anonymous requests have a username of system:anonymous, and a group name of system:unauthenticated.
	Enabled *bool `json:"enabled"`
}
