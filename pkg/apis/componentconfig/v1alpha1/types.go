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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ClientConnectionConfiguration contains details for constructing a client.
type ClientConnectionConfiguration struct {
	// kubeConfigFile is the path to a kubeconfig file.
	KubeConfigFile string `json:"kubeconfig"`
	// acceptContentTypes defines the Accept header sent by clients when connecting to a server, overriding the
	// default value of 'application/json'. This field will control all connections to the server used by a particular
	// client.
	AcceptContentTypes string `json:"acceptContentTypes"`
	// contentType is the content type used when sending data to the server from this client.
	ContentType string `json:"contentType"`
	// cps controls the number of queries per second allowed for this connection.
	QPS float32 `json:"qps"`
	// burst allows extra queries to accumulate when a client is exceeding its rate.
	Burst int `json:"burst"`
}

// KubeProxyIPTablesConfiguration contains iptables-related configuration
// details for the Kubernetes proxy server.
type KubeProxyIPTablesConfiguration struct {
	// masqueradeBit is the bit of the iptables fwmark space to use for SNAT if using
	// the pure iptables proxy mode. Values must be within the range [0, 31].
	MasqueradeBit *int32 `json:"masqueradeBit"`
	// masqueradeAll tells kube-proxy to SNAT everything if using the pure iptables proxy mode.
	MasqueradeAll bool `json:"masqueradeAll"`
	// syncPeriod is the period that iptables rules are refreshed (e.g. '5s', '1m',
	// '2h22m').  Must be greater than 0.
	SyncPeriod metav1.Duration `json:"syncPeriod"`
	// minSyncPeriod is the minimum period that iptables rules are refreshed (e.g. '5s', '1m',
	// '2h22m').
	MinSyncPeriod metav1.Duration `json:"minSyncPeriod"`
}

// KubeProxyConntrackConfiguration contains conntrack settings for
// the Kubernetes proxy server.
type KubeProxyConntrackConfiguration struct {
	// max is the maximum number of NAT connections to track (0 to
	// leave as-is).  This takes precedence over conntrackMaxPerCore and conntrackMin.
	Max int32 `json:"max"`
	// maxPerCore is the maximum number of NAT connections to track
	// per CPU core (0 to leave the limit as-is and ignore conntrackMin).
	MaxPerCore int32 `json:"maxPerCore"`
	// min is the minimum value of connect-tracking records to allocate,
	// regardless of conntrackMaxPerCore (set conntrackMaxPerCore=0 to leave the limit as-is).
	Min int32 `json:"min"`
	// tcpEstablishedTimeout is how long an idle TCP connection will be kept open
	// (e.g. '2s').  Must be greater than 0.
	TCPEstablishedTimeout metav1.Duration `json:"tcpEstablishedTimeout"`
	// tcpCloseWaitTimeout is how long an idle conntrack entry
	// in CLOSE_WAIT state will remain in the conntrack
	// table. (e.g. '60s'). Must be greater than 0 to set.
	TCPCloseWaitTimeout metav1.Duration `json:"tcpCloseWaitTimeout"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeProxyConfiguration contains everything necessary to configure the
// Kubernetes proxy server.
type KubeProxyConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// featureGates is a comma-separated list of key=value pairs that control
	// which alpha/beta features are enabled.
	//
	// TODO this really should be a map but that requires refactoring all
	// components to use config files because local-up-cluster.sh only supports
	// the --feature-gates flag right now, which is comma-separated key=value
	// pairs.
	FeatureGates string `json:"featureGates"`

	// bindAddress is the IP address for the proxy server to serve on (set to 0.0.0.0
	// for all interfaces)
	BindAddress string `json:"bindAddress"`
	// healthzBindAddress is the IP address and port for the health check server to serve on,
	// defaulting to 0.0.0.0:10256
	HealthzBindAddress string `json:"healthzBindAddress"`
	// metricsBindAddress is the IP address and port for the metrics server to serve on,
	// defaulting to 127.0.0.1:10249 (set to 0.0.0.0 for all interfaces)
	MetricsBindAddress string `json:"metricsBindAddress"`
	// enableProfiling enables profiling via web interface on /debug/pprof handler.
	// Profiling handlers will be handled by metrics server.
	EnableProfiling bool `json:"enableProfiling"`
	// clusterCIDR is the CIDR range of the pods in the cluster. It is used to
	// bridge traffic coming from outside of the cluster. If not provided,
	// no off-cluster bridging will be performed.
	ClusterCIDR string `json:"clusterCIDR"`
	// hostnameOverride, if non-empty, will be used as the identity instead of the actual hostname.
	HostnameOverride string `json:"hostnameOverride"`
	// clientConnection specifies the kubeconfig file and client connection settings for the proxy
	// server to use when communicating with the apiserver.
	ClientConnection ClientConnectionConfiguration `json:"clientConnection"`
	// iptables contains iptables-related configuration options.
	IPTables KubeProxyIPTablesConfiguration `json:"iptables"`
	// oomScoreAdj is the oom-score-adj value for kube-proxy process. Values must be within
	// the range [-1000, 1000]
	OOMScoreAdj *int32 `json:"oomScoreAdj"`
	// mode specifies which proxy mode to use.
	Mode ProxyMode `json:"mode"`
	// portRange is the range of host ports (beginPort-endPort, inclusive) that may be consumed
	// in order to proxy service traffic. If unspecified (0-0) then ports will be randomly chosen.
	PortRange string `json:"portRange"`
	// resourceContainer is the bsolute name of the resource-only container to create and run
	// the Kube-proxy in (Default: /kube-proxy).
	ResourceContainer string `json:"resourceContainer"`
	// udpIdleTimeout is how long an idle UDP connection will be kept open (e.g. '250ms', '2s').
	// Must be greater than 0. Only applicable for proxyMode=userspace.
	UDPIdleTimeout metav1.Duration `json:"udpTimeoutMilliseconds"`
	// conntrack contains conntrack-related configuration options.
	Conntrack KubeProxyConntrackConfiguration `json:"conntrack"`
	// configSyncPeriod is how often configuration from the apiserver is refreshed. Must be greater
	// than 0.
	ConfigSyncPeriod metav1.Duration `json:"configSyncPeriod"`
}

// Currently two modes of proxying are available: 'userspace' (older, stable) or 'iptables'
// (newer, faster). If blank, use the best-available proxy (currently iptables, but may
// change in future versions).  If the iptables proxy is selected, regardless of how, but
// the system's kernel or iptables versions are insufficient, this always falls back to the
// userspace proxy.
type ProxyMode string

const (
	ProxyModeUserspace ProxyMode = "userspace"
	ProxyModeIPTables  ProxyMode = "iptables"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type KubeSchedulerConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// port is the port that the scheduler's http service runs on.
	Port int `json:"port"`
	// address is the IP address to serve on.
	Address string `json:"address"`
	// algorithmProvider is the scheduling algorithm provider to use.
	AlgorithmProvider string `json:"algorithmProvider"`
	// policyConfigFile is the filepath to the scheduler policy configuration.
	PolicyConfigFile string `json:"policyConfigFile"`
	// enableProfiling enables profiling via web interface.
	EnableProfiling *bool `json:"enableProfiling"`
	// enableContentionProfiling enables lock contention profiling, if enableProfiling is true.
	EnableContentionProfiling bool `json:"enableContentionProfiling"`
	// contentType is contentType of requests sent to apiserver.
	ContentType string `json:"contentType"`
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver.
	KubeAPIQPS float32 `json:"kubeAPIQPS"`
	// kubeAPIBurst is the QPS burst to use while talking with kubernetes apiserver.
	KubeAPIBurst int `json:"kubeAPIBurst"`
	// schedulerName is name of the scheduler, used to select which pods
	// will be processed by this scheduler, based on pod's "spec.SchedulerName".
	SchedulerName string `json:"schedulerName"`
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	HardPodAffinitySymmetricWeight int `json:"hardPodAffinitySymmetricWeight"`
	// Indicate the "all topologies" set for empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.
	FailureDomains string `json:"failureDomains"`
	// leaderElection defines the configuration of leader election client.
	LeaderElection LeaderElectionConfiguration `json:"leaderElection"`
	// LockObjectNamespace defines the namespace of the lock object
	LockObjectNamespace string `json:"lockObjectNamespace"`
	// LockObjectName defines the lock object name
	LockObjectName string `json:"lockObjectName"`
	// PolicyConfigMapName is the name of the ConfigMap object that specifies
	// the scheduler's policy config. If UseLegacyPolicyConfig is true, scheduler
	// uses PolicyConfigFile. If UseLegacyPolicyConfig is false and
	// PolicyConfigMapName is not empty, the ConfigMap object with this name must
	// exist in PolicyConfigMapNamespace before scheduler initialization.
	PolicyConfigMapName string `json:"policyConfigMapName"`
	// PolicyConfigMapNamespace is the namespace where the above policy config map
	// is located. If none is provided default system namespace ("kube-system")
	// will be used.
	PolicyConfigMapNamespace string `json:"policyConfigMapNamespace"`
	// UseLegacyPolicyConfig tells the scheduler to ignore Policy ConfigMap and
	// to use PolicyConfigFile if available.
	UseLegacyPolicyConfig bool `json:"useLegacyPolicyConfig"`
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

// LeaderElectionConfiguration defines the configuration of leader election
// clients for components that can run with leader election enabled.
type LeaderElectionConfiguration struct {
	// leaderElect enables a leader election client to gain leadership
	// before executing the main loop. Enable this when running replicated
	// components for high availability.
	LeaderElect *bool `json:"leaderElect"`
	// leaseDuration is the duration that non-leader candidates will wait
	// after observing a leadership renewal until attempting to acquire
	// leadership of a led but unrenewed leader slot. This is effectively the
	// maximum duration that a leader can be stopped before it is replaced
	// by another candidate. This is only applicable if leader election is
	// enabled.
	LeaseDuration metav1.Duration `json:"leaseDuration"`
	// renewDeadline is the interval between attempts by the acting master to
	// renew a leadership slot before it stops leading. This must be less
	// than or equal to the lease duration. This is only applicable if leader
	// election is enabled.
	RenewDeadline metav1.Duration `json:"renewDeadline"`
	// retryPeriod is the duration the clients should wait between attempting
	// acquisition and renewal of a leadership. This is only applicable if
	// leader election is enabled.
	RetryPeriod metav1.Duration `json:"retryPeriod"`
	// resourceLock indicates the resource object type that will be used to lock
	// during leader election cycles.
	ResourceLock string `json:"resourceLock"`
}

const (
	// "kube-system" is the default scheduler lock object namespace
	SchedulerDefaultLockObjectNamespace string = "kube-system"

	// "kube-scheduler" is the default scheduler lock object name
	SchedulerDefaultLockObjectName = "kube-scheduler"
)
