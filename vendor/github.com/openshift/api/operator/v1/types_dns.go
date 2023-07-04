package v1

import (
	v1 "github.com/openshift/api/config/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
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

	// cache describes the caching configuration that applies to all server blocks listed in the Corefile.
	// This field allows a cluster admin to optionally configure:
	// * positiveTTL which is a duration for which positive responses should be cached.
	// * negativeTTL which is a duration for which negative responses should be cached.
	// If this is not configured, OpenShift will configure positive and negative caching with a default value that is
	// subject to change. At the time of writing, the default positiveTTL is 900 seconds and the default negativeTTL is
	// 30 seconds or as noted in the respective Corefile for your version of OpenShift.
	// +optional
	Cache DNSCache `json:"cache,omitempty"`
}

// DNSCache defines the fields for configuring DNS caching.
type DNSCache struct {
	// positiveTTL is optional and specifies the amount of time that a positive response should be cached.
	//
	// If configured, it must be a value of 1s (1 second) or greater up to a theoretical maximum of several years. This
	// field expects an unsigned duration string of decimal numbers, each with optional fraction and a unit suffix,
	// e.g. "100s", "1m30s", "12h30m10s". Values that are fractions of a second are rounded down to the nearest second.
	// If the configured value is less than 1s, the default value will be used.
	// If not configured, the value will be 0s and OpenShift will use a default value of 900 seconds unless noted
	// otherwise in the respective Corefile for your version of OpenShift. The default value of 900 seconds is subject
	// to change.
	// +kubebuilder:validation:Pattern=^(0|([0-9]+(\.[0-9]+)?(ns|us|µs|μs|ms|s|m|h))+)$
	// +kubebuilder:validation:Type:=string
	// +optional
	PositiveTTL metav1.Duration `json:"positiveTTL,omitempty"`

	// negativeTTL is optional and specifies the amount of time that a negative response should be cached.
	//
	// If configured, it must be a value of 1s (1 second) or greater up to a theoretical maximum of several years. This
	// field expects an unsigned duration string of decimal numbers, each with optional fraction and a unit suffix,
	// e.g. "100s", "1m30s", "12h30m10s". Values that are fractions of a second are rounded down to the nearest second.
	// If the configured value is less than 1s, the default value will be used.
	// If not configured, the value will be 0s and OpenShift will use a default value of 30 seconds unless noted
	// otherwise in the respective Corefile for your version of OpenShift. The default value of 30 seconds is subject
	// to change.
	// +kubebuilder:validation:Pattern=^(0|([0-9]+(\.[0-9]+)?(ns|us|µs|μs|ms|s|m|h))+)$
	// +kubebuilder:validation:Type:=string
	// +optional
	NegativeTTL metav1.Duration `json:"negativeTTL,omitempty"`
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

// DNSTransport indicates what type of connection should be used.
// +kubebuilder:validation:Enum=TLS;Cleartext;""
type DNSTransport string

const (
	// TLSTransport indicates that TLS should be used for the connection.
	TLSTransport DNSTransport = "TLS"

	// CleartextTransport indicates that no encryption should be used for
	// the connection.
	CleartextTransport DNSTransport = "Cleartext"
)

// DNSTransportConfig groups related configuration parameters used for configuring
// forwarding to upstream resolvers that support DNS-over-TLS.
// +union
type DNSTransportConfig struct {
	// transport allows cluster administrators to opt-in to using a DNS-over-TLS
	// connection between cluster DNS and an upstream resolver(s). Configuring
	// TLS as the transport at this level without configuring a CABundle will
	// result in the system certificates being used to verify the serving
	// certificate of the upstream resolver(s).
	//
	// Possible values:
	// "" (empty) - This means no explicit choice has been made and the platform chooses the default which is subject
	// to change over time. The current default is "Cleartext".
	// "Cleartext" - Cluster admin specified cleartext option. This results in the same functionality
	// as an empty value but may be useful when a cluster admin wants to be more explicit about the transport,
	// or wants to switch from "TLS" to "Cleartext" explicitly.
	// "TLS" - This indicates that DNS queries should be sent over a TLS connection. If Transport is set to TLS,
	// you MUST also set ServerName. If a port is not included with the upstream IP, port 853 will be tried by default
	// per RFC 7858 section 3.1; https://datatracker.ietf.org/doc/html/rfc7858#section-3.1.
	//
	// +optional
	// +unionDiscriminator
	Transport DNSTransport `json:"transport,omitempty"`

	// tls contains the additional configuration options to use when Transport is set to "TLS".
	TLS *DNSOverTLSConfig `json:"tls,omitempty"`
}

// DNSOverTLSConfig describes optional DNSTransportConfig fields that should be captured.
type DNSOverTLSConfig struct {
	// serverName is the upstream server to connect to when forwarding DNS queries. This is required when Transport is
	// set to "TLS". ServerName will be validated against the DNS naming conventions in RFC 1123 and should match the
	// TLS certificate installed in the upstream resolver(s).
	//
	// + ---
	// + Inspired by the DNS1123 patterns in Kubernetes: https://github.com/kubernetes/kubernetes/blob/7c46f40bdf89a437ecdbc01df45e235b5f6d9745/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L178-L218
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:Pattern=`^([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])(\.([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]{0,61}[a-zA-Z0-9]))*$`
	ServerName string `json:"serverName"`

	// caBundle references a ConfigMap that must contain either a single
	// CA Certificate or a CA Bundle. This allows cluster administrators to provide their
	// own CA or CA bundle for validating the certificate of upstream resolvers.
	//
	// 1. The configmap must contain a `ca-bundle.crt` key.
	// 2. The value must be a PEM encoded CA certificate or CA bundle.
	// 3. The administrator must create this configmap in the openshift-config namespace.
	// 4. The upstream server certificate must contain a Subject Alternative Name (SAN) that matches ServerName.
	//
	// +optional
	CABundle v1.ConfigMapNameReference `json:"caBundle,omitempty"`
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

	// transportConfig is used to configure the transport type, server name, and optional custom CA or CA bundle to use
	// when forwarding DNS requests to an upstream resolver.
	//
	// The default value is "" (empty) which results in a standard cleartext connection being used when forwarding DNS
	// requests to an upstream resolver.
	//
	// +optional
	TransportConfig DNSTransportConfig `json:"transportConfig,omitempty"`


	// protocolStrategy specifies the protocol to use for upstream DNS
	// requests.
	// Valid values for protocolStrategy are "TCP" and omitted.
	// When omitted, this means no opinion and the platform is left to choose
	// a reasonable default, which is subject to change over time.
	// The current default is to use the protocol of the original client request.
	// "TCP" specifies that the platform should use TCP for all upstream DNS requests,
	// even if the client request uses UDP.
	// "TCP" is useful for UDP-specific issues such as those created by
	// non-compliant upstream resolvers, but may consume more bandwidth or
	// increase DNS response time. Note that protocolStrategy only affects
	// the protocol of DNS requests that CoreDNS makes to upstream resolvers.
	// It does not affect the protocol of DNS requests between clients and
	// CoreDNS.
	//
	// +optional
	ProtocolStrategy ProtocolStrategy `json:"protocolStrategy"`
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

	// transportConfig is used to configure the transport type, server name, and optional custom CA or CA bundle to use
	// when forwarding DNS requests to an upstream resolver.
	//
	// The default value is "" (empty) which results in a standard cleartext connection being used when forwarding DNS
	// requests to an upstream resolver.
	//
	// +optional
	TransportConfig DNSTransportConfig `json:"transportConfig,omitempty"`

	// protocolStrategy specifies the protocol to use for upstream DNS
	// requests.
	// Valid values for protocolStrategy are "TCP" and omitted.
	// When omitted, this means no opinion and the platform is left to choose
	// a reasonable default, which is subject to change over time.
	// The current default is to use the protocol of the original client request.
	// "TCP" specifies that the platform should use TCP for all upstream DNS requests,
	// even if the client request uses UDP.
	// "TCP" is useful for UDP-specific issues such as those created by
	// non-compliant upstream resolvers, but may consume more bandwidth or
	// increase DNS response time. Note that protocolStrategy only affects
	// the protocol of DNS requests that CoreDNS makes to upstream resolvers.
	// It does not affect the protocol of DNS requests between clients and
	// CoreDNS.
	//
	// +optional
	ProtocolStrategy ProtocolStrategy `json:"protocolStrategy"`
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

// ProtocolStrategy is a preference for the protocol to use for DNS queries.
// + ---
// + When consumers observe an unknown value, they should use the default strategy.
// +kubebuilder:validation:Enum:=TCP;""
type ProtocolStrategy string

var (
	// ProtocolStrategyDefault specifies no opinion for DNS protocol.
	// If empty, the default behavior of CoreDNS is used. Currently, this means that CoreDNS uses the protocol of the
	// originating client request as the upstream protocol.
	// Note that the default behavior of CoreDNS is subject to change.
	ProtocolStrategyDefault ProtocolStrategy = ""

	// ProtocolStrategyTCP instructs CoreDNS to always use TCP, regardless of the originating client's request protocol.
	ProtocolStrategyTCP ProtocolStrategy = "TCP"
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

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []DNS `json:"items"`
}
