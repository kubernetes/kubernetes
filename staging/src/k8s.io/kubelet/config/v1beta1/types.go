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

package v1beta1

import (
	v1 "k8s.io/api/core/v1"
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

// ResourceChangeDetectionStrategy denotes a mode in which internal
// managers (secret, configmap) are discovering object changes.
type ResourceChangeDetectionStrategy string

// Enum settings for different strategies of kubelet managers.
const (
	// GetChangeDetectionStrategy is a mode in which kubelet fetches
	// necessary objects directly from apiserver.
	GetChangeDetectionStrategy ResourceChangeDetectionStrategy = "Get"
	// TTLCacheChangeDetectionStrategy is a mode in which kubelet uses
	// ttl cache for object directly fetched from apiserver.
	TTLCacheChangeDetectionStrategy ResourceChangeDetectionStrategy = "Cache"
	// WatchChangeDetectionStrategy is a mode in which kubelet uses
	// watches to observe changes to objects that are in its interest.
	WatchChangeDetectionStrategy ResourceChangeDetectionStrategy = "Watch"
	// RestrictedTopologyManagerPolicy is a mode in which kubelet only allows
	// pods with optimal NUMA node alignment for requested resources
	RestrictedTopologyManagerPolicy = "restricted"
	// BestEffortTopologyManagerPolicy is a mode in which kubelet will favour
	// pods with NUMA alignment of CPU and device resources.
	BestEffortTopologyManagerPolicy = "best-effort"
	// NoneTopologyManager Policy is a mode in which kubelet has no knowledge
	// of NUMA alignment of a pod's CPU and device resources.
	NoneTopologyManagerPolicy = "none"
	// SingleNumaNodeTopologyManager Policy iis a mode in which kubelet only allows
	// pods with a single NUMA alignment of CPU and device resources.
	SingleNumaNodeTopologyManager = "single-numa-node"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeletConfiguration contains the configuration for the Kubelet
type KubeletConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// enableServer enables Kubelet's secured server.
	// Note: Kubelet's insecure port is controlled by the readOnlyPort option.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Default: true
	EnableServer *bool `json:"enableServer,omitempty"`
	// staticPodPath is the path to the directory containing local (static) pods to
	// run, or the path to a single static pod file.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// the set of static pods specified at the new path may be different than the
	// ones the Kubelet initially started with, and this may disrupt your node.
	// Default: ""
	// +optional
	StaticPodPath string `json:"staticPodPath,omitempty"`
	// syncFrequency is the max period between synchronizing running
	// containers and config.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// shortening this duration may have a negative performance impact, especially
	// as the number of Pods on the node increases. Alternatively, increasing this
	// duration will result in longer refresh times for ConfigMaps and Secrets.
	// Default: "1m"
	// +optional
	SyncFrequency metav1.Duration `json:"syncFrequency,omitempty"`
	// fileCheckFrequency is the duration between checking config files for
	// new data
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// shortening the duration will cause the Kubelet to reload local Static Pod
	// configurations more frequently, which may have a negative performance impact.
	// Default: "20s"
	// +optional
	FileCheckFrequency metav1.Duration `json:"fileCheckFrequency,omitempty"`
	// httpCheckFrequency is the duration between checking http for new data
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// shortening the duration will cause the Kubelet to poll staticPodURL more
	// frequently, which may have a negative performance impact.
	// Default: "20s"
	// +optional
	HTTPCheckFrequency metav1.Duration `json:"httpCheckFrequency,omitempty"`
	// staticPodURL is the URL for accessing static pods to run
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// the set of static pods specified at the new URL may be different than the
	// ones the Kubelet initially started with, and this may disrupt your node.
	// Default: ""
	// +optional
	StaticPodURL string `json:"staticPodURL,omitempty"`
	// staticPodURLHeader is a map of slices with HTTP headers to use when accessing the podURL
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt the ability to read the latest set of static pods from StaticPodURL.
	// Default: nil
	// +optional
	StaticPodURLHeader map[string][]string `json:"staticPodURLHeader,omitempty"`
	// address is the IP address for the Kubelet to serve on (set to 0.0.0.0
	// for all interfaces).
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Default: "0.0.0.0"
	// +optional
	Address string `json:"address,omitempty"`
	// port is the port for the Kubelet to serve on.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Default: 10250
	// +optional
	Port int32 `json:"port,omitempty"`
	// readOnlyPort is the read-only port for the Kubelet to serve on with
	// no authentication/authorization.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Default: 0 (disabled)
	// +optional
	ReadOnlyPort int32 `json:"readOnlyPort,omitempty"`
	// tlsCertFile is the file containing x509 Certificate for HTTPS. (CA cert,
	// if any, concatenated after server cert). If tlsCertFile and
	// tlsPrivateKeyFile are not provided, a self-signed certificate
	// and key are generated for the public address and saved to the directory
	// passed to the Kubelet's --cert-dir flag.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Default: ""
	// +optional
	TLSCertFile string `json:"tlsCertFile,omitempty"`
	// tlsPrivateKeyFile is the file containing x509 private key matching tlsCertFile
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Default: ""
	// +optional
	TLSPrivateKeyFile string `json:"tlsPrivateKeyFile,omitempty"`
	// TLSCipherSuites is the list of allowed cipher suites for the server.
	// Values are from tls package constants (https://golang.org/pkg/crypto/tls/#pkg-constants).
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Default: nil
	// +optional
	TLSCipherSuites []string `json:"tlsCipherSuites,omitempty"`
	// TLSMinVersion is the minimum TLS version supported.
	// Values are from tls package constants (https://golang.org/pkg/crypto/tls/#pkg-constants).
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Default: ""
	// +optional
	TLSMinVersion string `json:"tlsMinVersion,omitempty"`
	// rotateCertificates enables client certificate rotation. The Kubelet will request a
	// new certificate from the certificates.k8s.io API. This requires an approver to approve the
	// certificate signing requests.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// disabling it may disrupt the Kubelet's ability to authenticate with the API server
	// after the current certificate expires.
	// Default: false
	// +optional
	RotateCertificates bool `json:"rotateCertificates,omitempty"`
	// serverTLSBootstrap enables server certificate bootstrap. Instead of self
	// signing a serving certificate, the Kubelet will request a certificate from
	// the certificates.k8s.io API. This requires an approver to approve the
	// certificate signing requests. The RotateKubeletServerCertificate feature
	// must be enabled.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// disabling it will stop the renewal of Kubelet server certificates, which can
	// disrupt components that interact with the Kubelet server in the long term,
	// due to certificate expiration.
	// Default: false
	// +optional
	ServerTLSBootstrap bool `json:"serverTLSBootstrap,omitempty"`
	// authentication specifies how requests to the Kubelet's server are authenticated
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Defaults:
	//   anonymous:
	//     enabled: false
	//   webhook:
	//     enabled: true
	//     cacheTTL: "2m"
	// +optional
	Authentication KubeletAuthentication `json:"authentication"`
	// authorization specifies how requests to the Kubelet's server are authorized
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Defaults:
	//   mode: Webhook
	//   webhook:
	//     cacheAuthorizedTTL: "5m"
	//     cacheUnauthorizedTTL: "30s"
	// +optional
	Authorization KubeletAuthorization `json:"authorization"`
	// registryPullQPS is the limit of registry pulls per second.
	// Set to 0 for no limit.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact scalability by changing the amount of traffic produced
	// by image pulls.
	// Default: 5
	// +optional
	RegistryPullQPS *int32 `json:"registryPullQPS,omitempty"`
	// registryBurst is the maximum size of bursty pulls, temporarily allows
	// pulls to burst to this number, while still not exceeding registryPullQPS.
	// Only used if registryPullQPS > 0.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact scalability by changing the amount of traffic produced
	// by image pulls.
	// Default: 10
	// +optional
	RegistryBurst int32 `json:"registryBurst,omitempty"`
	// eventRecordQPS is the maximum event creations per second. If 0, there
	// is no limit enforced.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact scalability by changing the amount of traffic produced by
	// event creations.
	// Default: 5
	// +optional
	EventRecordQPS *int32 `json:"eventRecordQPS,omitempty"`
	// eventBurst is the maximum size of a burst of event creations, temporarily
	// allows event creations to burst to this number, while still not exceeding
	// eventRecordQPS. Only used if eventRecordQPS > 0.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact scalability by changing the amount of traffic produced by
	// event creations.
	// Default: 10
	// +optional
	EventBurst int32 `json:"eventBurst,omitempty"`
	// enableDebuggingHandlers enables server endpoints for log access
	// and local running of containers and commands, including the exec,
	// attach, logs, and portforward features.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// disabling it may disrupt components that interact with the Kubelet server.
	// Default: true
	// +optional
	EnableDebuggingHandlers *bool `json:"enableDebuggingHandlers,omitempty"`
	// enableContentionProfiling enables lock contention profiling, if enableDebuggingHandlers is true.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// enabling it may carry a performance impact.
	// Default: false
	// +optional
	EnableContentionProfiling bool `json:"enableContentionProfiling,omitempty"`
	// healthzPort is the port of the localhost healthz endpoint (set to 0 to disable)
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that monitor Kubelet health.
	// Default: 10248
	// +optional
	HealthzPort *int32 `json:"healthzPort,omitempty"`
	// healthzBindAddress is the IP address for the healthz server to serve on
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that monitor Kubelet health.
	// Default: "127.0.0.1"
	// +optional
	HealthzBindAddress string `json:"healthzBindAddress,omitempty"`
	// oomScoreAdj is The oom-score-adj value for kubelet process. Values
	// must be within the range [-1000, 1000].
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact the stability of nodes under memory pressure.
	// Default: -999
	// +optional
	OOMScoreAdj *int32 `json:"oomScoreAdj,omitempty"`
	// clusterDomain is the DNS domain for this cluster. If set, kubelet will
	// configure all containers to search this domain in addition to the
	// host's search domains.
	// Dynamic Kubelet Config (beta): Dynamically updating this field is not recommended,
	// as it should be kept in sync with the rest of the cluster.
	// Default: ""
	// +optional
	ClusterDomain string `json:"clusterDomain,omitempty"`
	// clusterDNS is a list of IP addresses for the cluster DNS server. If set,
	// kubelet will configure all containers to use this for DNS resolution
	// instead of the host's DNS servers.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// changes will only take effect on Pods created after the update. Draining
	// the node is recommended before changing this field.
	// Default: nil
	// +optional
	ClusterDNS []string `json:"clusterDNS,omitempty"`
	// streamingConnectionIdleTimeout is the maximum time a streaming connection
	// can be idle before the connection is automatically closed.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact components that rely on infrequent updates over streaming
	// connections to the Kubelet server.
	// Default: "4h"
	// +optional
	StreamingConnectionIdleTimeout metav1.Duration `json:"streamingConnectionIdleTimeout,omitempty"`
	// nodeStatusUpdateFrequency is the frequency that kubelet computes node
	// status. If node lease feature is not enabled, it is also the frequency that
	// kubelet posts node status to master.
	// Note: When node lease feature is not enabled, be cautious when changing the
	// constant, it must work with nodeMonitorGracePeriod in nodecontroller.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact node scalability, and also that the node controller's
	// nodeMonitorGracePeriod must be set to N*NodeStatusUpdateFrequency,
	// where N is the number of retries before the node controller marks
	// the node unhealthy.
	// Default: "10s"
	// +optional
	NodeStatusUpdateFrequency metav1.Duration `json:"nodeStatusUpdateFrequency,omitempty"`
	// nodeStatusReportFrequency is the frequency that kubelet posts node
	// status to master if node status does not change. Kubelet will ignore this
	// frequency and post node status immediately if any change is detected. It is
	// only used when node lease feature is enabled. nodeStatusReportFrequency's
	// default value is 1m. But if nodeStatusUpdateFrequency is set explicitly,
	// nodeStatusReportFrequency's default value will be set to
	// nodeStatusUpdateFrequency for backward compatibility.
	// Default: "1m"
	// +optional
	NodeStatusReportFrequency metav1.Duration `json:"nodeStatusReportFrequency,omitempty"`
	// nodeLeaseDurationSeconds is the duration the Kubelet will set on its corresponding Lease,
	// when the NodeLease feature is enabled. This feature provides an indicator of node
	// health by having the Kubelet create and periodically renew a lease, named after the node,
	// in the kube-node-lease namespace. If the lease expires, the node can be considered unhealthy.
	// The lease is currently renewed every 10s, per KEP-0009. In the future, the lease renewal interval
	// may be set based on the lease duration.
	// Requires the NodeLease feature gate to be enabled.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// decreasing the duration may reduce tolerance for issues that temporarily prevent
	// the Kubelet from renewing the lease (e.g. a short-lived network issue).
	// Default: 40
	// +optional
	NodeLeaseDurationSeconds int32 `json:"nodeLeaseDurationSeconds,omitempty"`
	// imageMinimumGCAge is the minimum age for an unused image before it is
	// garbage collected.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may trigger or delay garbage collection, and may change the image overhead
	// on the node.
	// Default: "2m"
	// +optional
	ImageMinimumGCAge metav1.Duration `json:"imageMinimumGCAge,omitempty"`
	// imageGCHighThresholdPercent is the percent of disk usage after which
	// image garbage collection is always run. The percent is calculated as
	// this field value out of 100.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may trigger or delay garbage collection, and may change the image overhead
	// on the node.
	// Default: 85
	// +optional
	ImageGCHighThresholdPercent *int32 `json:"imageGCHighThresholdPercent,omitempty"`
	// imageGCLowThresholdPercent is the percent of disk usage before which
	// image garbage collection is never run. Lowest disk usage to garbage
	// collect to. The percent is calculated as this field value out of 100.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may trigger or delay garbage collection, and may change the image overhead
	// on the node.
	// Default: 80
	// +optional
	ImageGCLowThresholdPercent *int32 `json:"imageGCLowThresholdPercent,omitempty"`
	// How frequently to calculate and cache volume disk usage for all pods
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// shortening the period may carry a performance impact.
	// Default: "1m"
	// +optional
	VolumeStatsAggPeriod metav1.Duration `json:"volumeStatsAggPeriod,omitempty"`
	// kubeletCgroups is the absolute name of cgroups to isolate the kubelet in
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: ""
	// +optional
	KubeletCgroups string `json:"kubeletCgroups,omitempty"`
	// systemCgroups is absolute name of cgroups in which to place
	// all non-kernel processes that are not already in a container. Empty
	// for no container. Rolling back the flag requires a reboot.
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: ""
	// +optional
	SystemCgroups string `json:"systemCgroups,omitempty"`
	// cgroupRoot is the root cgroup to use for pods. This is handled by the
	// container runtime on a best effort basis.
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: ""
	// +optional
	CgroupRoot string `json:"cgroupRoot,omitempty"`
	// Enable QoS based Cgroup hierarchy: top level cgroups for QoS Classes
	// And all Burstable and BestEffort pods are brought up under their
	// specific top level QoS cgroup.
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: true
	// +optional
	CgroupsPerQOS *bool `json:"cgroupsPerQOS,omitempty"`
	// driver that the kubelet uses to manipulate cgroups on the host (cgroupfs or systemd)
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: "cgroupfs"
	// +optional
	CgroupDriver string `json:"cgroupDriver,omitempty"`
	// CPUManagerPolicy is the name of the policy to use.
	// Requires the CPUManager feature gate to be enabled.
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: "none"
	// +optional
	CPUManagerPolicy string `json:"cpuManagerPolicy,omitempty"`
	// CPU Manager reconciliation period.
	// Requires the CPUManager feature gate to be enabled.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// shortening the period may carry a performance impact.
	// Default: "10s"
	// +optional
	CPUManagerReconcilePeriod metav1.Duration `json:"cpuManagerReconcilePeriod,omitempty"`
	// TopologyManagerPolicy is the name of the policy to use.
	// Policies other than "none" require the TopologyManager feature gate to be enabled.
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: "none"
	// +optional
	TopologyManagerPolicy string `json:"topologyManagerPolicy,omitempty"`
	// qosReserved is a set of resource name to percentage pairs that specify
	// the minimum percentage of a resource reserved for exclusive use by the
	// guaranteed QoS tier.
	// Currently supported resources: "memory"
	// Requires the QOSReserved feature gate to be enabled.
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: nil
	// +optional
	QOSReserved map[string]string `json:"qosReserved,omitempty"`
	// runtimeRequestTimeout is the timeout for all runtime requests except long running
	// requests - pull, logs, exec and attach.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may disrupt components that interact with the Kubelet server.
	// Default: "2m"
	// +optional
	RuntimeRequestTimeout metav1.Duration `json:"runtimeRequestTimeout,omitempty"`
	// hairpinMode specifies how the Kubelet should configure the container
	// bridge for hairpin packets.
	// Setting this flag allows endpoints in a Service to loadbalance back to
	// themselves if they should try to access their own Service. Values:
	//   "promiscuous-bridge": make the container bridge promiscuous.
	//   "hairpin-veth":       set the hairpin flag on container veth interfaces.
	//   "none":               do nothing.
	// Generally, one must set --hairpin-mode=hairpin-veth to achieve hairpin NAT,
	// because promiscuous-bridge assumes the existence of a container bridge named cbr0.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may require a node reboot, depending on the network plugin.
	// Default: "promiscuous-bridge"
	// +optional
	HairpinMode string `json:"hairpinMode,omitempty"`
	// maxPods is the number of pods that can run on this Kubelet.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// changes may cause Pods to fail admission on Kubelet restart, and may change
	// the value reported in Node.Status.Capacity[v1.ResourcePods], thus affecting
	// future scheduling decisions. Increasing this value may also decrease performance,
	// as more Pods can be packed into a single node.
	// Default: 110
	// +optional
	MaxPods int32 `json:"maxPods,omitempty"`
	// The CIDR to use for pod IP addresses, only used in standalone mode.
	// In cluster mode, this is obtained from the master.
	// Dynamic Kubelet Config (beta): This field should always be set to the empty default.
	// It should only set for standalone Kubelets, which cannot use Dynamic Kubelet Config.
	// Default: ""
	// +optional
	PodCIDR string `json:"podCIDR,omitempty"`
	// PodPidsLimit is the maximum number of pids in any pod.
	// Requires the SupportPodPidsLimit feature gate to be enabled.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// lowering it may prevent container processes from forking after the change.
	// Default: -1
	// +optional
	PodPidsLimit *int64 `json:"podPidsLimit,omitempty"`
	// ResolverConfig is the resolver configuration file used as the basis
	// for the container DNS resolution configuration.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// changes will only take effect on Pods created after the update. Draining
	// the node is recommended before changing this field.
	// Default: "/etc/resolv.conf"
	// +optional
	ResolverConfig string `json:"resolvConf,omitempty"`
	// RunOnce causes the Kubelet to check the API server once for pods,
	// run those in addition to the pods specified by static pod files, and exit.
	// Default: false
	// +optional
	RunOnce bool `json:"runOnce,omitempty"`
	// cpuCFSQuota enables CPU CFS quota enforcement for containers that
	// specify CPU limits.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// disabling it may reduce node stability.
	// Default: true
	// +optional
	CPUCFSQuota *bool `json:"cpuCFSQuota,omitempty"`
	// CPUCFSQuotaPeriod is the CPU CFS quota period value, cpu.cfs_period_us.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// limits set for containers will result in different cpu.cfs_quota settings. This
	// will trigger container restarts on the node being reconfigured.
	// Default: "100ms"
	// +optional
	CPUCFSQuotaPeriod *metav1.Duration `json:"cpuCFSQuotaPeriod,omitempty"`
	// nodeStatusMaxImages caps the number of images reported in Node.Status.Images.
	// Note: If -1 is specified, no cap will be applied. If 0 is specified, no image is returned.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// different values can be reported on node status.
	// Default: 50
	// +optional
	NodeStatusMaxImages *int32 `json:"nodeStatusMaxImages,omitempty"`
	// maxOpenFiles is Number of files that can be opened by Kubelet process.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact the ability of the Kubelet to interact with the node's filesystem.
	// Default: 1000000
	// +optional
	MaxOpenFiles int64 `json:"maxOpenFiles,omitempty"`
	// contentType is contentType of requests sent to apiserver.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact the ability for the Kubelet to communicate with the API server.
	// If the Kubelet loses contact with the API server due to a change to this field,
	// the change cannot be reverted via dynamic Kubelet config.
	// Default: "application/vnd.kubernetes.protobuf"
	// +optional
	ContentType string `json:"contentType,omitempty"`
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact scalability by changing the amount of traffic the Kubelet
	// sends to the API server.
	// Default: 5
	// +optional
	KubeAPIQPS *int32 `json:"kubeAPIQPS,omitempty"`
	// kubeAPIBurst is the burst to allow while talking with kubernetes apiserver
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact scalability by changing the amount of traffic the Kubelet
	// sends to the API server.
	// Default: 10
	// +optional
	KubeAPIBurst int32 `json:"kubeAPIBurst,omitempty"`
	// serializeImagePulls when enabled, tells the Kubelet to pull images one
	// at a time. We recommend *not* changing the default value on nodes that
	// run docker daemon with version  < 1.9 or an Aufs storage backend.
	// Issue #10959 has more details.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact the performance of image pulls.
	// Default: true
	// +optional
	SerializeImagePulls *bool `json:"serializeImagePulls,omitempty"`
	// Map of signal names to quantities that defines hard eviction thresholds. For example: {"memory.available": "300Mi"}.
	// To explicitly disable, pass a 0% or 100% threshold on an arbitrary resource.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may trigger or delay Pod evictions.
	// Default:
	//   memory.available:  "100Mi"
	//   nodefs.available:  "10%"
	//   nodefs.inodesFree: "5%"
	//   imagefs.available: "15%"
	// +optional
	EvictionHard map[string]string `json:"evictionHard,omitempty"`
	// Map of signal names to quantities that defines soft eviction thresholds.
	// For example: {"memory.available": "300Mi"}.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may trigger or delay Pod evictions, and may change the allocatable reported
	// by the node.
	// Default: nil
	// +optional
	EvictionSoft map[string]string `json:"evictionSoft,omitempty"`
	// Map of signal names to quantities that defines grace periods for each soft eviction signal.
	// For example: {"memory.available": "30s"}.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may trigger or delay Pod evictions.
	// Default: nil
	// +optional
	EvictionSoftGracePeriod map[string]string `json:"evictionSoftGracePeriod,omitempty"`
	// Duration for which the kubelet has to wait before transitioning out of an eviction pressure condition.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// lowering it may decrease the stability of the node when the node is overcommitted.
	// Default: "5m"
	// +optional
	EvictionPressureTransitionPeriod metav1.Duration `json:"evictionPressureTransitionPeriod,omitempty"`
	// Maximum allowed grace period (in seconds) to use when terminating pods in
	// response to a soft eviction threshold being met. This value effectively caps
	// the Pod's TerminationGracePeriodSeconds value during soft evictions.
	// Note: Due to issue #64530, the behavior has a bug where this value currently just
	// overrides the grace period during soft eviction, which can increase the grace
	// period from what is set on the Pod. This bug will be fixed in a future release.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// lowering it decreases the amount of time Pods will have to gracefully clean
	// up before being killed during a soft eviction.
	// Default: 0
	// +optional
	EvictionMaxPodGracePeriod int32 `json:"evictionMaxPodGracePeriod,omitempty"`
	// Map of signal names to quantities that defines minimum reclaims, which describe the minimum
	// amount of a given resource the kubelet will reclaim when performing a pod eviction while
	// that resource is under pressure. For example: {"imagefs.available": "2Gi"}
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may change how well eviction can manage resource pressure.
	// Default: nil
	// +optional
	EvictionMinimumReclaim map[string]string `json:"evictionMinimumReclaim,omitempty"`
	// podsPerCore is the maximum number of pods per core. Cannot exceed MaxPods.
	// If 0, this field is ignored.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// changes may cause Pods to fail admission on Kubelet restart, and may change
	// the value reported in Node.Status.Capacity[v1.ResourcePods], thus affecting
	// future scheduling decisions. Increasing this value may also decrease performance,
	// as more Pods can be packed into a single node.
	// Default: 0
	// +optional
	PodsPerCore int32 `json:"podsPerCore,omitempty"`
	// enableControllerAttachDetach enables the Attach/Detach controller to
	// manage attachment/detachment of volumes scheduled to this node, and
	// disables kubelet from executing any attach/detach operations
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// changing which component is responsible for volume management on a live node
	// may result in volumes refusing to detach if the node is not drained prior to
	// the update, and if Pods are scheduled to the node before the
	// volumes.kubernetes.io/controller-managed-attach-detach annotation is updated by the
	// Kubelet. In general, it is safest to leave this value set the same as local config.
	// Default: true
	// +optional
	EnableControllerAttachDetach *bool `json:"enableControllerAttachDetach,omitempty"`
	// protectKernelDefaults, if true, causes the Kubelet to error if kernel
	// flags are not as it expects. Otherwise the Kubelet will attempt to modify
	// kernel flags to match its expectation.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// enabling it may cause the Kubelet to crash-loop if the Kernel is not configured as
	// Kubelet expects.
	// Default: false
	// +optional
	ProtectKernelDefaults bool `json:"protectKernelDefaults,omitempty"`
	// If true, Kubelet ensures a set of iptables rules are present on host.
	// These rules will serve as utility rules for various components, e.g. KubeProxy.
	// The rules will be created based on IPTablesMasqueradeBit and IPTablesDropBit.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// disabling it will prevent the Kubelet from healing locally misconfigured iptables rules.
	// Default: true
	// +optional
	MakeIPTablesUtilChains *bool `json:"makeIPTablesUtilChains,omitempty"`
	// iptablesMasqueradeBit is the bit of the iptables fwmark space to mark for SNAT
	// Values must be within the range [0, 31]. Must be different from other mark bits.
	// Warning: Please match the value of the corresponding parameter in kube-proxy.
	// TODO: clean up IPTablesMasqueradeBit in kube-proxy
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it needs to be coordinated with other components, like kube-proxy, and the update
	// will only be effective if MakeIPTablesUtilChains is enabled.
	// Default: 14
	// +optional
	IPTablesMasqueradeBit *int32 `json:"iptablesMasqueradeBit,omitempty"`
	// iptablesDropBit is the bit of the iptables fwmark space to mark for dropping packets.
	// Values must be within the range [0, 31]. Must be different from other mark bits.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it needs to be coordinated with other components, like kube-proxy, and the update
	// will only be effective if MakeIPTablesUtilChains is enabled.
	// Default: 15
	// +optional
	IPTablesDropBit *int32 `json:"iptablesDropBit,omitempty"`
	// featureGates is a map of feature names to bools that enable or disable alpha/experimental
	// features. This field modifies piecemeal the built-in default values from
	// "k8s.io/kubernetes/pkg/features/kube_features.go".
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider the
	// documentation for the features you are enabling or disabling. While we
	// encourage feature developers to make it possible to dynamically enable
	// and disable features, some changes may require node reboots, and some
	// features may require careful coordination to retroactively disable.
	// Default: nil
	// +optional
	FeatureGates map[string]bool `json:"featureGates,omitempty"`
	// failSwapOn tells the Kubelet to fail to start if swap is enabled on the node.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// setting it to true will cause the Kubelet to crash-loop if swap is enabled.
	// Default: true
	// +optional
	FailSwapOn *bool `json:"failSwapOn,omitempty"`
	// A quantity defines the maximum size of the container log file before it is rotated.
	// For example: "5Mi" or "256Ki".
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may trigger log rotation.
	// Default: "10Mi"
	// +optional
	ContainerLogMaxSize string `json:"containerLogMaxSize,omitempty"`
	// Maximum number of container log files that can be present for a container.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// lowering it may cause log files to be deleted.
	// Default: 5
	// +optional
	ContainerLogMaxFiles *int32 `json:"containerLogMaxFiles,omitempty"`
	// ConfigMapAndSecretChangeDetectionStrategy is a mode in which
	// config map and secret managers are running.
	// Default: "Watch"
	// +optional
	ConfigMapAndSecretChangeDetectionStrategy ResourceChangeDetectionStrategy `json:"configMapAndSecretChangeDetectionStrategy,omitempty"`

	/* the following fields are meant for Node Allocatable */

	// systemReserved is a set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G)
	// pairs that describe resources reserved for non-kubernetes components.
	// Currently only cpu and memory are supported.
	// See http://kubernetes.io/docs/user-guide/compute-resources for more detail.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may not be possible to increase the reserved resources, because this
	// requires resizing cgroups. Always look for a NodeAllocatableEnforced event
	// after updating this field to ensure that the update was successful.
	// Default: nil
	// +optional
	SystemReserved map[string]string `json:"systemReserved,omitempty"`
	// A set of ResourceName=ResourceQuantity (e.g. cpu=200m,memory=150G) pairs
	// that describe resources reserved for kubernetes system components.
	// Currently cpu, memory and local storage for root file system are supported.
	// See http://kubernetes.io/docs/user-guide/compute-resources for more detail.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may not be possible to increase the reserved resources, because this
	// requires resizing cgroups. Always look for a NodeAllocatableEnforced event
	// after updating this field to ensure that the update was successful.
	// Default: nil
	// +optional
	KubeReserved map[string]string `json:"kubeReserved,omitempty"`
	// This ReservedSystemCPUs option specifies the cpu list reserved for the host level system threads and kubernetes related threads.
	// This provide a "static" CPU list rather than the "dynamic" list by system-reserved and kube-reserved.
	// This option overwrites CPUs provided by system-reserved and kube-reserved.
	ReservedSystemCPUs string `json:"reservedSystemCPUs,omitempty"`
	// The previous version for which you want to show hidden metrics.
	// Only the previous minor version is meaningful, other values will not be allowed.
	// The format is <major>.<minor>, e.g.: '1.16'.
	// The purpose of this format is make sure you have the opportunity to notice if the next release hides additional metrics,
	// rather than being surprised when they are permanently removed in the release after that.
	// Default: ""
	// +optional
	ShowHiddenMetricsForVersion string `json:"showHiddenMetricsForVersion,omitempty"`
	// This flag helps kubelet identify absolute name of top level cgroup used to enforce `SystemReserved` compute resource reservation for OS system daemons.
	// Refer to [Node Allocatable](https://git.k8s.io/community/contributors/design-proposals/node/node-allocatable.md) doc for more information.
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: ""
	// +optional
	SystemReservedCgroup string `json:"systemReservedCgroup,omitempty"`
	// This flag helps kubelet identify absolute name of top level cgroup used to enforce `KubeReserved` compute resource reservation for Kubernetes node system daemons.
	// Refer to [Node Allocatable](https://git.k8s.io/community/contributors/design-proposals/node/node-allocatable.md) doc for more information.
	// Dynamic Kubelet Config (beta): This field should not be updated without a full node
	// reboot. It is safest to keep this value the same as the local config.
	// Default: ""
	// +optional
	KubeReservedCgroup string `json:"kubeReservedCgroup,omitempty"`
	// This flag specifies the various Node Allocatable enforcements that Kubelet needs to perform.
	// This flag accepts a list of options. Acceptable options are `none`, `pods`, `system-reserved` & `kube-reserved`.
	// If `none` is specified, no other options may be specified.
	// Refer to [Node Allocatable](https://git.k8s.io/community/contributors/design-proposals/node/node-allocatable.md) doc for more information.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// removing enforcements may reduce the stability of the node. Alternatively, adding
	// enforcements may reduce the stability of components which were using more than
	// the reserved amount of resources; for example, enforcing kube-reserved may cause
	// Kubelets to OOM if it uses more than the reserved resources, and enforcing system-reserved
	// may cause system daemons to OOM if they use more than the reserved resources.
	// Default: ["pods"]
	// +optional
	EnforceNodeAllocatable []string `json:"enforceNodeAllocatable,omitempty"`
	// A comma separated whitelist of unsafe sysctls or sysctl patterns (ending in *).
	// Unsafe sysctl groups are kernel.shm*, kernel.msg*, kernel.sem, fs.mqueue.*, and net.*.
	// These sysctls are namespaced but not allowed by default.  For example: "kernel.msg*,net.ipv4.route.min_pmtu"
	// Default: []
	// +optional
	AllowedUnsafeSysctls []string `json:"allowedUnsafeSysctls,omitempty"`
	// volumePluginDir is the full path of the directory in which to search
	// for additional third party volume plugins.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that changing
	// the volumePluginDir may disrupt workloads relying on third party volume plugins.
	// Default: "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/"
	// +optional
	VolumePluginDir string `json:"volumePluginDir,omitempty"`
	// providerID, if set, sets the unique id of the instance that an external provider (i.e. cloudprovider)
	// can use to identify a specific node.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact the ability of the Kubelet to interact with cloud providers.
	// Default: ""
	// +optional
	ProviderID string `json:"providerID,omitempty"`
	// kernelMemcgNotification, if set, the kubelet will integrate with the kernel memcg notification
	// to determine if memory eviction thresholds are crossed rather than polling.
	// Dynamic Kubelet Config (beta): If dynamically updating this field, consider that
	// it may impact the way Kubelet interacts with the kernel.
	// Default: false
	// +optional
	KernelMemcgNotification bool `json:"kernelMemcgNotification,omitempty"`
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
	// +optional
	Mode KubeletAuthorizationMode `json:"mode,omitempty"`

	// webhook contains settings related to Webhook authorization.
	// +optional
	Webhook KubeletWebhookAuthorization `json:"webhook"`
}

type KubeletWebhookAuthorization struct {
	// cacheAuthorizedTTL is the duration to cache 'authorized' responses from the webhook authorizer.
	// +optional
	CacheAuthorizedTTL metav1.Duration `json:"cacheAuthorizedTTL,omitempty"`
	// cacheUnauthorizedTTL is the duration to cache 'unauthorized' responses from the webhook authorizer.
	// +optional
	CacheUnauthorizedTTL metav1.Duration `json:"cacheUnauthorizedTTL,omitempty"`
}

type KubeletAuthentication struct {
	// x509 contains settings related to x509 client certificate authentication
	// +optional
	X509 KubeletX509Authentication `json:"x509"`
	// webhook contains settings related to webhook bearer token authentication
	// +optional
	Webhook KubeletWebhookAuthentication `json:"webhook"`
	// anonymous contains settings related to anonymous authentication
	// +optional
	Anonymous KubeletAnonymousAuthentication `json:"anonymous"`
}

type KubeletX509Authentication struct {
	// clientCAFile is the path to a PEM-encoded certificate bundle. If set, any request presenting a client certificate
	// signed by one of the authorities in the bundle is authenticated with a username corresponding to the CommonName,
	// and groups corresponding to the Organization in the client certificate.
	// +optional
	ClientCAFile string `json:"clientCAFile,omitempty"`
}

type KubeletWebhookAuthentication struct {
	// enabled allows bearer token authentication backed by the tokenreviews.authentication.k8s.io API
	// +optional
	Enabled *bool `json:"enabled,omitempty"`
	// cacheTTL enables caching of authentication results
	// +optional
	CacheTTL metav1.Duration `json:"cacheTTL,omitempty"`
}

type KubeletAnonymousAuthentication struct {
	// enabled allows anonymous requests to the kubelet server.
	// Requests that are not rejected by another authentication method are treated as anonymous requests.
	// Anonymous requests have a username of system:anonymous, and a group name of system:unauthenticated.
	// +optional
	Enabled *bool `json:"enabled,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SerializedNodeConfigSource allows us to serialize v1.NodeConfigSource.
// This type is used internally by the Kubelet for tracking checkpointed dynamic configs.
// It exists in the kubeletconfig API group because it is classified as a versioned input to the Kubelet.
type SerializedNodeConfigSource struct {
	metav1.TypeMeta `json:",inline"`
	// Source is the source that we are serializing
	// +optional
	Source v1.NodeConfigSource `json:"source,omitempty" protobuf:"bytes,1,opt,name=source"`
}
