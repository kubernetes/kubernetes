package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	corev1 "k8s.io/api/core/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=dnses,scope=Cluster
// +kubebuilder:subresource:status

// DNS manages the CoreDNS component to provide a name resolution service
// for pods and services in the cluster.
//
// This supports the DNS-based service discovery specification:
// https://github.com/kubernetes/dns/blob/master/docs/specification.md
//
// More details: https://kubernetes.io/docs/tasks/administer-cluster/coredns
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type DNS struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec is the specification of the desired behavior of the DNS.
	Spec DNSSpec `json:"spec,omitempty"`
	// status is the most recently observed status of the DNS.
	Status DNSStatus `json:"status,omitempty"`
}

// DNSSpec is the specification of the desired behavior of the DNS.
type DNSSpec struct {
	// servers is a list of DNS resolvers that provide name query delegation for one or
	// more subdomains outside the scope of the cluster domain. If servers consists of
	// more than one Server, longest suffix match will be used to determine the Server.
	//
	// For example, if there are two Servers, one for "foo.com" and another for "a.foo.com",
	// and the name query is for "www.a.foo.com", it will be routed to the Server with Zone
	// "a.foo.com".
	//
	// If this field is nil, no servers are created.
	//
	// +optional
	Servers []Server `json:"servers,omitempty"`

	// upstreamResolvers defines a schema for configuring CoreDNS
	// to proxy DNS messages to upstream resolvers for the case of the
	// default (".") server
	//
	// If this field is not specified, the upstream used will default to
	// /etc/resolv.conf, with policy "sequential"
	//
	// +optional
	UpstreamResolvers UpstreamResolvers `json:"upstreamResolvers"`

	// nodePlacement provides explicit control over the scheduling of DNS
	// pods.
	//
	// Generally, it is useful to run a DNS pod on every node so that DNS
	// queries are always handled by a local DNS pod instead of going over
	// the network to a DNS pod on another node.  However, security policies
	// may require restricting the placement of DNS pods to specific nodes.
	// For example, if a security policy prohibits pods on arbitrary nodes
	// from communicating with the API, a node selector can be specified to
	// restrict DNS pods to nodes that are permitted to communicate with the
	// API.  Conversely, if running DNS pods on nodes with a particular
	// taint is desired, a toleration can be specified for that taint.
	//
	// If unset, defaults are used. See nodePlacement for more details.
	//
	// +optional
	NodePlacement DNSNodePlacement `json:"nodePlacement,omitempty"`

	// managementState indicates whether the DNS operator should manage cluster
	// DNS
	// +optional
	ManagementState ManagementState `json:"managementState,omitempty"`

	// operatorLogLevel controls the logging level of the DNS Operator.
	// Valid values are: "Normal", "Debug", "Trace".
	// Defaults to "Normal".
	// setting operatorLogLevel: Trace will produce extremely verbose logs.
	// +optional
	// +kubebuilder:default=Normal
	OperatorLogLevel DNSLogLevel `json:"operatorLogLevel,omitempty"`

	// logLevel describes the desired logging verbosity for CoreDNS.
	// Any one of the following values may be specified:
	// * Normal logs errors from upstream resolvers.
	// * Debug logs errors, NXDOMAIN responses, and NODATA responses.
	// * Trace logs errors and all responses.
	//  Setting logLevel: Trace will produce extremely verbose logs.
	// Valid values are: "Normal", "Debug", "Trace".
	// Defaults to "Normal".
	// +optional
	// +kubebuilder:default=Normal
	LogLevel DNSLogLevel `json:"logLevel,omitempty"`
}

// +kubebuilder:validation:Enum:=Normal;Debug;Trace
type DNSLogLevel string

var (
	// Normal is the default.  Normal, working log information, everything is fine, but helpful notices for auditing or common operations.  In kube, this is probably glog=2.
	DNSLogLevelNormal DNSLogLevel = "Normal"

	// Debug is used when something went wrong.  Even common operations may be logged, and less helpful but more quantity of notices.  In kube, this is probably glog=4.
	DNSLogLevelDebug DNSLogLevel = "Debug"

	// Trace is used when something went really badly and even more verbose logs are needed.  Logging every function call as part of a common operation, to tracing execution of a query.  In kube, this is probably glog=6.
	DNSLogLevelTrace DNSLogLevel = "Trace"
)

// Server defines the schema for a server that runs per instance of CoreDNS.
type Server struct {
	// name is required and specifies a unique name for the server. Name must comply
	// with the Service Name Syntax of rfc6335.
	Name string `json:"name"`
	// zones is required and specifies the subdomains that Server is authoritative for.
	// Zones must conform to the rfc1123 definition of a subdomain. Specifying the
	// cluster domain (i.e., "cluster.local") is invalid.
	Zones []string `json:"zones"`
	// forwardPlugin defines a schema for configuring CoreDNS to proxy DNS messages
	// to upstream resolvers.
	ForwardPlugin ForwardPlugin `json:"forwardPlugin"`
}

// ForwardingPolicy is the policy to use when forwarding DNS requests.
// +kubebuilder:validation:Enum=Random;RoundRobin;Sequential
type ForwardingPolicy string

const (
	// RandomForwardingPolicy picks a random upstream server for each query.
	RandomForwardingPolicy ForwardingPolicy = "Random"

	// RoundRobinForwardingPolicy picks upstream servers in a round-robin order, moving to the next server for each new query.
	RoundRobinForwardingPolicy ForwardingPolicy = "RoundRobin"

	// SequentialForwardingPolicy tries querying upstream servers in a sequential order until one responds, starting with the first server for each new query.
	SequentialForwardingPolicy ForwardingPolicy = "Sequential"
)

// ForwardPlugin defines a schema for configuring the CoreDNS forward plugin.
type ForwardPlugin struct {
	// upstreams is a list of resolvers to forward name queries for subdomains of Zones.
	// Each instance of CoreDNS performs health checking of Upstreams. When a healthy upstream
	// returns an error during the exchange, another resolver is tried from Upstreams. The
	// Upstreams are selected in the order specified in Policy. Each upstream is represented
	// by an IP address or IP:port if the upstream listens on a port other than 53.
	//
	// A maximum of 15 upstreams is allowed per ForwardPlugin.
	//
	// +kubebuilder:validation:MaxItems=15
	Upstreams []string `json:"upstreams"`

	// policy is used to determine the order in which upstream servers are selected for querying.
	// Any one of the following values may be specified:
	//
	// * "Random" picks a random upstream server for each query.
	// * "RoundRobin" picks upstream servers in a round-robin order, moving to the next server for each new query.
	// * "Sequential" tries querying upstream servers in a sequential order until one responds, starting with the first server for each new query.
	//
	// The default value is "Random"
	//
	// +optional
	// +kubebuilder:default:="Random"
	Policy ForwardingPolicy `json:"policy,omitempty"`
}

// UpstreamResolvers defines a schema for configuring the CoreDNS forward plugin in the
// specific case of the default (".") server.
// It defers from ForwardPlugin in the default values it accepts:
// * At least one upstream should be specified.
// * the default policy is Sequential
type UpstreamResolvers struct {
	// Upstreams is a list of resolvers to forward name queries for the "." domain.
	// Each instance of CoreDNS performs health checking of Upstreams. When a healthy upstream
	// returns an error during the exchange, another resolver is tried from Upstreams. The
	// Upstreams are selected in the order specified in Policy.
	//
	// A maximum of 15 upstreams is allowed per ForwardPlugin.
	// If no Upstreams are specified, /etc/resolv.conf is used by default
	//
	// +optional
	// +kubebuilder:validation:MaxItems=15
	// +kubebuilder:default={{"type":"SystemResolvConf"}}
	Upstreams []Upstream `json:"upstreams"`

	// Policy is used to determine the order in which upstream servers are selected for querying.
	// Any one of the following values may be specified:
	//
	// * "Random" picks a random upstream server for each query.
	// * "RoundRobin" picks upstream servers in a round-robin order, moving to the next server for each new query.
	// * "Sequential" tries querying upstream servers in a sequential order until one responds, starting with the first server for each new query.
	//
	// The default value is "Sequential"
	//
	// +optional
	// +kubebuilder:default="Sequential"
	Policy ForwardingPolicy `json:"policy,omitempty"`
}

// Upstream can either be of type SystemResolvConf, or of type Network.
//
// * For an Upstream of type SystemResolvConf, no further fields are necessary:
//   The upstream will be configured to use /etc/resolv.conf.
// * For an Upstream of type Network, a NetworkResolver field needs to be defined
//   with an IP address or IP:port if the upstream listens on a port other than 53.
type Upstream struct {

	// Type defines whether this upstream contains an IP/IP:port resolver or the local /etc/resolv.conf.
	// Type accepts 2 possible values: SystemResolvConf or Network.
	//
	// * When SystemResolvConf is used, the Upstream structure does not require any further fields to be defined:
	//   /etc/resolv.conf will be used
	// * When Network is used, the Upstream structure must contain at least an Address
	//
	// +kubebuilder:validation:Required
	// +required
	Type UpstreamType `json:"type"`

	// Address must be defined when Type is set to Network. It will be ignored otherwise.
	// It must be a valid ipv4 or ipv6 address.
	//
	// +optional
	// +kubebuilder:validation:Optional
	Address string `json:"address,omitempty"`

	// Port may be defined when Type is set to Network. It will be ignored otherwise.
	// Port must be between 65535
	//
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:validation:Optional
	// +kubebuilder:default=53
	Port uint32 `json:"port,omitempty"`
}

// +kubebuilder:validation:Enum=SystemResolvConf;Network;""
type UpstreamType string

const (
	SystemResolveConfType UpstreamType = "SystemResolvConf"
	NetworkResolverType   UpstreamType = "Network"
)

// DNSNodePlacement describes the node scheduling configuration for DNS pods.
type DNSNodePlacement struct {
	// nodeSelector is the node selector applied to DNS pods.
	//
	// If empty, the default is used, which is currently the following:
	//
	//   kubernetes.io/os: linux
	//
	// This default is subject to change.
	//
	// If set, the specified selector is used and replaces the default.
	//
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// tolerations is a list of tolerations applied to DNS pods.
	//
	// If empty, the DNS operator sets a toleration for the
	// "node-role.kubernetes.io/master" taint.  This default is subject to
	// change.  Specifying tolerations without including a toleration for
	// the "node-role.kubernetes.io/master" taint may be risky as it could
	// lead to an outage if all worker nodes become unavailable.
	//
	// Note that the daemon controller adds some tolerations as well.  See
	// https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/
	//
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`
}

const (
	// Available indicates the DNS controller daemonset is available.
	DNSAvailable = "Available"
)

// DNSStatus defines the observed status of the DNS.
type DNSStatus struct {
	// clusterIP is the service IP through which this DNS is made available.
	//
	// In the case of the default DNS, this will be a well known IP that is used
	// as the default nameserver for pods that are using the default ClusterFirst DNS policy.
	//
	// In general, this IP can be specified in a pod's spec.dnsConfig.nameservers list
	// or used explicitly when performing name resolution from within the cluster.
	// Example: dig foo.com @<service IP>
	//
	// More info: https://kubernetes.io/docs/concepts/services-networking/service/#virtual-ips-and-service-proxies
	//
	// +kubebuilder:validation:Required
	// +required
	ClusterIP string `json:"clusterIP"`

	// clusterDomain is the local cluster DNS domain suffix for DNS services.
	// This will be a subdomain as defined in RFC 1034,
	// section 3.5: https://tools.ietf.org/html/rfc1034#section-3.5
	// Example: "cluster.local"
	//
	// More info: https://kubernetes.io/docs/concepts/services-networking/dns-pod-service
	//
	// +kubebuilder:validation:Required
	// +required
	ClusterDomain string `json:"clusterDomain"`

	// conditions provide information about the state of the DNS on the cluster.
	//
	// These are the supported DNS conditions:
	//
	//   * Available
	//   - True if the following conditions are met:
	//     * DNS controller daemonset is available.
	//   - False if any of those conditions are unsatisfied.
	//
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +optional
	Conditions []OperatorCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true

// DNSList contains a list of DNS
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type DNSList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []DNS `json:"items"`
}
