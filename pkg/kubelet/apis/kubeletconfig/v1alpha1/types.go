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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// A configuration field should go in KubeletFlags instead of KubeletConfiguration if
// its value cannot be safely shared between nodes at the same time (e.g. a hostname)
// In general, please try to avoid adding flags or configuration fields,
// we already have a confusingly large amount of them.
type KubeletConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Only used for dynamic configuration.
	// The length of the trial period for this configuration. This configuration will become the last-known-good after this duration.
	ConfigTrialDuration *metav1.Duration `json:"configTrialDuration"`
	// podManifestPath is the path to the directory containing pod manifests to
	// run, or the path to a single manifest file
	PodManifestPath string `json:"podManifestPath"`
	// syncFrequency is the max period between synchronizing running
	// containers and config
	SyncFrequency metav1.Duration `json:"syncFrequency"`
	// fileCheckFrequency is the duration between checking config files for
	// new data
	FileCheckFrequency metav1.Duration `json:"fileCheckFrequency"`
	// httpCheckFrequency is the duration between checking http for new data
	HTTPCheckFrequency metav1.Duration `json:"httpCheckFrequency"`
	// manifestURL is the URL for accessing the container manifest
	ManifestURL string `json:"manifestURL"`
	// manifestURLHeader is the HTTP header to use when accessing the manifest
	// URL, with the key separated from the value with a ':', as in 'key:value'
	ManifestURLHeader map[string][]string `json:"manifestURLHeader"`
	// enableServer enables the Kubelet's server
	EnableServer *bool `json:"enableServer"`
	// address is the IP address for the Kubelet to serve on (set to 0.0.0.0
	// for all interfaces)
	Address string `json:"address"`
	// port is the port for the Kubelet to serve on.
	Port int32 `json:"port"`
	// readOnlyPort is the read-only port for the Kubelet to serve on with
	// no authentication/authorization (set to 0 to disable)
	ReadOnlyPort *int32 `json:"readOnlyPort"`
	// tlsCertFile is the file containing x509 Certificate for HTTPS.  (CA cert,
	// if any, concatenated after server cert). If tlsCertFile and
	// tlsPrivateKeyFile are not provided, a self-signed certificate
	// and key are generated for the public address and saved to the directory
	// passed to certDir.
	TLSCertFile string `json:"tlsCertFile"`
	// tlsPrivateKeyFile is the ile containing x509 private key matching
	// tlsCertFile.
	TLSPrivateKeyFile string `json:"tlsPrivateKeyFile"`
	// TLSCipherSuites is the list of allowed cipher suites for the server.
	// Values are from tls package constants (https://golang.org/pkg/crypto/tls/#pkg-constants).
	TLSCipherSuites []string `json:"tlsCipherSuites"`
	// TLSMinVersion is the minimum TLS version supported.
	// Values are from tls package constants (https://golang.org/pkg/crypto/tls/#pkg-constants).
	TLSMinVersion string `json:"tlsMinVersion"`
	// authentication specifies how requests to the Kubelet's server are authenticated
	Authentication KubeletAuthentication `json:"authentication"`
	// authorization specifies how requests to the Kubelet's server are authorized
	Authorization KubeletAuthorization `json:"authorization"`
	// registryPullQPS is the limit of registry pulls per second. If 0,
	// unlimited. Set to 0 for no limit. Defaults to 5.0.
	RegistryPullQPS *int32 `json:"registryPullQPS"`
	// registryBurst is the maximum size of a bursty pulls, temporarily allows
	// pulls to burst to this number, while still not exceeding registryQps.
	// Only used if registryQPS > 0.
	RegistryBurst int32 `json:"registryBurst"`
	// eventRecordQPS is the maximum event creations per second. If 0, there
	// is no limit enforced.
	EventRecordQPS *int32 `json:"eventRecordQPS"`
	// eventBurst is the maximum size of a bursty event records, temporarily
	// allows event records to burst to this number, while still not exceeding
	// event-qps. Only used if eventQps > 0
	EventBurst int32 `json:"eventBurst"`
	// enableDebuggingHandlers enables server endpoints for log collection
	// and local running of containers and commands
	EnableDebuggingHandlers *bool `json:"enableDebuggingHandlers"`
	// enableContentionProfiling enables lock contention profiling, if enableDebuggingHandlers is true.
	EnableContentionProfiling bool `json:"enableContentionProfiling"`
	// cAdvisorPort is the port of the localhost cAdvisor endpoint (set to 0 to disable)
	CAdvisorPort *int32 `json:"cAdvisorPort"`
	// healthzPort is the port of the localhost healthz endpoint (set to 0 to disable)
	HealthzPort *int32 `json:"healthzPort"`
	// healthzBindAddress is the IP address for the healthz server to serve
	// on.
	HealthzBindAddress string `json:"healthzBindAddress"`
	// oomScoreAdj is The oom-score-adj value for kubelet process. Values
	// must be within the range [-1000, 1000].
	OOMScoreAdj *int32 `json:"oomScoreAdj"`
	// clusterDomain is the DNS domain for this cluster. If set, kubelet will
	// configure all containers to search this domain in addition to the
	// host's search domains.
	ClusterDomain string `json:"clusterDomain"`
	// clusterDNS is a list of IP address for the cluster DNS server.  If set,
	// kubelet will configure all containers to use this for DNS resolution
	// instead of the host's DNS servers
	ClusterDNS []string `json:"clusterDNS"`
	// streamingConnectionIdleTimeout is the maximum time a streaming connection
	// can be idle before the connection is automatically closed.
	StreamingConnectionIdleTimeout metav1.Duration `json:"streamingConnectionIdleTimeout"`
	// nodeStatusUpdateFrequency is the frequency that kubelet posts node
	// status to master. Note: be cautious when changing the constant, it
	// must work with nodeMonitorGracePeriod in nodecontroller.
	NodeStatusUpdateFrequency metav1.Duration `json:"nodeStatusUpdateFrequency"`
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
	// How frequently to calculate and cache volume disk usage for all pods
	VolumeStatsAggPeriod metav1.Duration `json:"volumeStatsAggPeriod"`
	// kubeletCgroups is the absolute name of cgroups to isolate the kubelet in.
	KubeletCgroups string `json:"kubeletCgroups"`
	// systemCgroups is absolute name of cgroups in which to place
	// all non-kernel processes that are not already in a container. Empty
	// for no container. Rolling back the flag requires a reboot.
	SystemCgroups string `json:"systemCgroups"`
	// cgroupRoot is the root cgroup to use for pods. This is handled by the
	// container runtime on a best effort basis.
	CgroupRoot string `json:"cgroupRoot"`
	// Enable QoS based Cgroup hierarchy: top level cgroups for QoS Classes
	// And all Burstable and BestEffort pods are brought up under their
	// specific top level QoS cgroup.
	// +optional
	CgroupsPerQOS *bool `json:"cgroupsPerQOS,omitempty"`
	// driver that the kubelet uses to manipulate cgroups on the host (cgroupfs or systemd)
	// +optional
	CgroupDriver string `json:"cgroupDriver,omitempty"`
	// CPUManagerPolicy is the name of the policy to use.
	CPUManagerPolicy string `json:"cpuManagerPolicy"`
	// CPU Manager reconciliation period.
	CPUManagerReconcilePeriod metav1.Duration `json:"cpuManagerReconcilePeriod"`
	// runtimeRequestTimeout is the timeout for all runtime requests except long running
	// requests - pull, logs, exec and attach.
	RuntimeRequestTimeout metav1.Duration `json:"runtimeRequestTimeout"`
	// How should the kubelet configure the container bridge for hairpin packets.
	// Setting this flag allows endpoints in a Service to loadbalance back to
	// themselves if they should try to access their own Service. Values:
	//   "promiscuous-bridge": make the container bridge promiscuous.
	//   "hairpin-veth":       set the hairpin flag on container veth interfaces.
	//   "none":               do nothing.
	// Generally, one must set --hairpin-mode=hairpin-veth to achieve hairpin NAT,
	// because promiscous-bridge assumes the existence of a container bridge named cbr0.
	HairpinMode string `json:"hairpinMode"`
	// maxPods is the number of pods that can run on this Kubelet.
	MaxPods int32 `json:"maxPods"`
	// The CIDR to use for pod IP addresses, only used in standalone mode.
	// In cluster mode, this is obtained from the master.
	PodCIDR string `json:"podCIDR"`
	// PodPidsLimit is the maximum number of pids in any pod.
	PodPidsLimit *int64 `json:"podPidsLimit"`
	// ResolverConfig is the resolver configuration file used as the basis
	// for the container DNS resolution configuration.
	ResolverConfig string `json:"resolvConf"`
	// cpuCFSQuota is Enable CPU CFS quota enforcement for containers that
	// specify CPU limits
	CPUCFSQuota *bool `json:"cpuCFSQuota"`
	// maxOpenFiles is Number of files that can be opened by Kubelet process.
	MaxOpenFiles int64 `json:"maxOpenFiles"`
	// contentType is contentType of requests sent to apiserver.
	ContentType string `json:"contentType"`
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver
	KubeAPIQPS *int32 `json:"kubeAPIQPS"`
	// kubeAPIBurst is the burst to allow while talking with kubernetes
	// apiserver
	KubeAPIBurst int32 `json:"kubeAPIBurst"`
	// serializeImagePulls when enabled, tells the Kubelet to pull images one
	// at a time. We recommend *not* changing the default value on nodes that
	// run docker daemon with version  < 1.9 or an Aufs storage backend.
	// Issue #10959 has more details.
	SerializeImagePulls *bool `json:"serializeImagePulls"`
	// Map of signal names to quantities that defines hard eviction thresholds. For example: {"memory.available": "300Mi"}.
	// +optional
	EvictionHard map[string]string `json:"evictionHard"`
	// Map of signal names to quantities that defines soft eviction thresholds.  For example: {"memory.available": "300Mi"}.
	// +optional
	EvictionSoft map[string]string `json:"evictionSoft"`
	// Map of signal names to quantities that defines grace periods for each soft eviction signal. For example: {"memory.available": "30s"}.
	// +optional
	EvictionSoftGracePeriod map[string]string `json:"evictionSoftGracePeriod"`
	// Duration for which the kubelet has to wait before transitioning out of an eviction pressure condition.
	EvictionPressureTransitionPeriod metav1.Duration `json:"evictionPressureTransitionPeriod"`
	// Maximum allowed grace period (in seconds) to use when terminating pods in response to a soft eviction threshold being met.
	EvictionMaxPodGracePeriod int32 `json:"evictionMaxPodGracePeriod"`
	// Map of signal names to quantities that defines minimum reclaims, which describe the minimum
	// amount of a given resource the kubelet will reclaim when performing a pod eviction while
	// that resource is under pressure. For example: {"imagefs.available": "2Gi"}
	// +optional
	EvictionMinimumReclaim map[string]string `json:"evictionMinimumReclaim"`
	// Maximum number of pods per core. Cannot exceed MaxPods
	PodsPerCore int32 `json:"podsPerCore"`
	// enableControllerAttachDetach enables the Attach/Detach controller to
	// manage attachment/detachment of volumes scheduled to this node, and
	// disables kubelet from executing any attach/detach operations
	EnableControllerAttachDetach *bool `json:"enableControllerAttachDetach"`
	// Default behaviour for kernel tuning
	ProtectKernelDefaults bool `json:"protectKernelDefaults"`
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
	// featureGates is a map of feature names to bools that enable or disable alpha/experimental features.
	FeatureGates map[string]bool `json:"featureGates,omitempty"`
	// Tells the Kubelet to fail to start if swap is enabled on the node.
	FailSwapOn *bool `json:"failSwapOn,omitempty"`

	/* following flags are meant for Node Allocatable */

	// A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs
	// that describe resources reserved for non-kubernetes components.
	// Currently only cpu and memory are supported. [default=none]
	// See http://kubernetes.io/docs/user-guide/compute-resources for more detail.
	SystemReserved map[string]string `json:"systemReserved"`
	// A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs
	// that describe resources reserved for kubernetes system components.
	// Currently cpu, memory and local storage for root file system are supported. [default=none]
	// See http://kubernetes.io/docs/user-guide/compute-resources for more detail.
	KubeReserved map[string]string `json:"kubeReserved"`

	// This flag helps kubelet identify absolute name of top level cgroup used to enforce `SystemReserved` compute resource reservation for OS system daemons.
	// Refer to [Node Allocatable](https://git.k8s.io/community/contributors/design-proposals/node/node-allocatable.md) doc for more information.
	SystemReservedCgroup string `json:"systemReservedCgroup,omitempty"`
	// This flag helps kubelet identify absolute name of top level cgroup used to enforce `KubeReserved` compute resource reservation for Kubernetes node system daemons.
	// Refer to [Node Allocatable](https://git.k8s.io/community/contributors/design-proposals/node/node-allocatable.md) doc for more information.
	KubeReservedCgroup string `json:"kubeReservedCgroup,omitempty"`
	// This flag specifies the various Node Allocatable enforcements that Kubelet needs to perform.
	// This flag accepts a list of options. Acceptible options are `pods`, `system-reserved` & `kube-reserved`.
	// Refer to [Node Allocatable](https://git.k8s.io/community/contributors/design-proposals/node/node-allocatable.md) doc for more information.
	EnforceNodeAllocatable []string `json:"enforceNodeAllocatable"`
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
