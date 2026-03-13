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

package networking

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NetworkPolicy describes what network traffic is allowed for a set of pods
type NetworkPolicy struct {
	metav1.TypeMeta

	// +optional
	metav1.ObjectMeta

	// spec represents the specification of the desired behavior for this NetworkPolicy.
	// +optional
	Spec NetworkPolicySpec
}

// PolicyType describes the NetworkPolicy type
// This type is beta-level in 1.8
type PolicyType string

const (
	// PolicyTypeIngress is a NetworkPolicy that affects ingress traffic on selected pods
	PolicyTypeIngress PolicyType = "Ingress"
	// PolicyTypeEgress is a NetworkPolicy that affects egress traffic on selected pods
	PolicyTypeEgress PolicyType = "Egress"
)

// NetworkPolicySpec provides the specification of a NetworkPolicy
type NetworkPolicySpec struct {
	// podSelector selects the pods to which this NetworkPolicy object applies.
	// The array of rules is applied to any pods selected by this field. An empty
	// selector matches all pods in the policy's namespace.
	// Multiple network policies can select the same set of pods. In this case,
	// the ingress rules for each are combined additively.
	// This field is optional. If it is not specified, it defaults to an empty selector.
	// +optional
	PodSelector metav1.LabelSelector

	// ingress is a list of ingress rules to be applied to the selected pods.
	// Traffic is allowed to a pod if there are no NetworkPolicies selecting the pod
	// (and cluster policy otherwise allows the traffic), OR if the traffic source is
	// the pod's local node, OR if the traffic matches at least one ingress rule
	// across all of the NetworkPolicy objects whose podSelector matches the pod. If
	// this field is empty then this NetworkPolicy does not allow any traffic (and serves
	// solely to ensure that the pods it selects are isolated by default)
	// +optional
	Ingress []NetworkPolicyIngressRule

	// egress is a list of egress rules to be applied to the selected pods. Outgoing traffic
	// is allowed if there are no NetworkPolicies selecting the pod (and cluster policy
	// otherwise allows the traffic), OR if the traffic matches at least one egress rule
	// across all of the NetworkPolicy objects whose podSelector matches the pod. If
	// this field is empty then this NetworkPolicy limits all outgoing traffic (and serves
	// solely to ensure that the pods it selects are isolated by default).
	// This field is beta-level in 1.8
	// +optional
	Egress []NetworkPolicyEgressRule

	// policyTypes is a list of rule types that the NetworkPolicy relates to.
	// Valid options are ["Ingress"], ["Egress"], or ["Ingress", "Egress"].
	// If this field is not specified, it will default based on the existence of ingress or egress rules;
	// policies that contain an egress section are assumed to affect egress, and all policies
	// (whether or not they contain an ingress section) are assumed to affect ingress.
	// If you want to write an egress-only policy, you must explicitly specify policyTypes [ "Egress" ].
	// Likewise, if you want to write a policy that specifies that no egress is allowed,
	// you must specify a policyTypes value that include "Egress" (since such a policy would not include
	// an egress section and would otherwise default to just [ "Ingress" ]).
	// This field is beta-level in 1.8
	// +optional
	PolicyTypes []PolicyType
}

// NetworkPolicyIngressRule describes a particular set of traffic that is allowed to the pods
// matched by a NetworkPolicySpec's podSelector. The traffic must match both ports and from.
type NetworkPolicyIngressRule struct {
	// ports is a list of ports which should be made accessible on the pods selected for
	// this rule. Each item in this list is combined using a logical OR. If this field is
	// empty or missing, this rule matches all ports (traffic not restricted by port).
	// If this field is present and contains at least one item, then this rule allows
	// traffic only if the traffic matches at least one port in the list.
	// +optional
	Ports []NetworkPolicyPort

	// from is a list of sources which should be able to access the pods selected for this rule.
	// Items in this list are combined using a logical OR operation. If this field is
	// empty or missing, this rule matches all sources (traffic not restricted by
	// source). If this field is present and contains at least one item, this rule
	// allows traffic only if the traffic matches at least one item in the from list.
	// +optional
	From []NetworkPolicyPeer
}

// NetworkPolicyEgressRule describes a particular set of traffic that is allowed out of pods
// matched by a NetworkPolicySpec's podSelector. The traffic must match both ports and to.
// This type is beta-level in 1.8
type NetworkPolicyEgressRule struct {
	// ports is a list of destination ports for outgoing traffic.
	// Each item in this list is combined using a logical OR. If this field is
	// empty or missing, this rule matches all ports (traffic not restricted by port).
	// If this field is present and contains at least one item, then this rule allows
	// traffic only if the traffic matches at least one port in the list.
	// +optional
	Ports []NetworkPolicyPort

	// to is a list of destinations for outgoing traffic of pods selected for this rule.
	// Items in this list are combined using a logical OR operation. If this field is
	// empty or missing, this rule matches all destinations (traffic not restricted by
	// destination). If this field is present and contains at least one item, this rule
	// allows traffic only if the traffic matches at least one item in the to list.
	// +optional
	To []NetworkPolicyPeer
}

// NetworkPolicyPort describes a port to allow traffic on
type NetworkPolicyPort struct {
	// protocol represents the protocol (TCP, UDP, or SCTP) which traffic must match.
	// If not specified, this field defaults to TCP.
	// +optional
	Protocol *api.Protocol

	// port represents the port on the given protocol. This can either be a numerical or named
	// port on a pod. If this field is not provided, this matches all port names and
	// numbers.
	// If present, only traffic on the specified protocol AND port will be matched.
	// +optional
	Port *intstr.IntOrString

	// endPort indicates that the range of ports from port to endPort if set, inclusive,
	// should be allowed by the policy. This field cannot be defined if the port field
	// is not defined or if the port field is defined as a named (string) port.
	// The endPort must be equal or greater than port.
	// +optional
	EndPort *int32
}

// IPBlock describes a particular CIDR (Ex. "192.168.1.0/24","2001:db8::/64") that is allowed
// to the pods matched by a NetworkPolicySpec's podSelector. The except entry describes CIDRs
// that should not be included within this rule.
type IPBlock struct {
	// cidr is a string representing the IPBlock
	// Valid examples are "192.168.1.0/24" or "2001:db8::/64"
	CIDR string

	// except is a list of CIDRs that should not be included within the IPBlock
	// Valid examples are "192.168.1.0/24" or "2001:db8::/64"
	// Except values will be rejected if they are outside the cidr range
	// +optional
	Except []string
}

// NetworkPolicyPeer describes a peer to allow traffic to/from.
type NetworkPolicyPeer struct {
	// podSelector is a label selector which selects pods. This field follows standard label
	// selector semantics; if present but empty, it selects all pods.
	//
	// If namespaceSelector is also set, then the NetworkPolicyPeer as a whole selects
	// the pods matching podSelector in the Namespaces selected by namespaceSelector.
	// Otherwise it selects the pods matching podSelector in the policy's own namespace.
	// +optional
	PodSelector *metav1.LabelSelector

	// namespaceSelector selects namespaces using cluster-scoped labels. This field follows
	// standard label selector semantics; if present but empty, it selects all namespaces.
	//
	// If podSelector is also set, then the NetworkPolicyPeer as a whole selects
	// the pods matching podSelector in the namespaces selected by namespaceSelector.
	// Otherwise it selects all pods in the namespaces selected by namespaceSelector.
	// +optional
	NamespaceSelector *metav1.LabelSelector

	// ipBlock defines policy on a particular IPBlock. If this field is set then
	// neither of the other fields can be.
	// +optional
	IPBlock *IPBlock
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NetworkPolicyList is a list of NetworkPolicy objects.
type NetworkPolicyList struct {
	metav1.TypeMeta

	// +optional
	metav1.ListMeta

	Items []NetworkPolicy
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Ingress is a collection of rules that allow inbound connections to reach the
// endpoints defined by a backend. An Ingress can be configured to give services
// externally-reachable urls, load balance traffic, terminate SSL, offer name
// based virtual hosting etc.
type Ingress struct {
	metav1.TypeMeta

	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta

	// spec is the desired state of the Ingress.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec IngressSpec

	// status is the current state of the Ingress.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status IngressStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IngressList is a collection of Ingress.
type IngressList struct {
	metav1.TypeMeta

	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// items is the list of Ingress.
	Items []Ingress
}

// IngressSpec describes the Ingress the user wishes to exist.
type IngressSpec struct {
	// ingressClassName is the name of the IngressClass cluster resource. The
	// associated IngressClass defines which controller will implement the
	// resource. This replaces the deprecated `kubernetes.io/ingress.class`
	// annotation. For backwards compatibility, when that annotation is set, it
	// must be given precedence over this field. The controller may emit a
	// warning if the field and annotation have different values.
	// Implementations of this API should ignore Ingresses without a class
	// specified. An IngressClass resource may be marked as default, which can
	// be used to set a default value for this field. For more information,
	// refer to the IngressClass documentation.
	// +optional
	IngressClassName *string

	// defaultBackend is the backend that should handle requests that don't
	// match any rule. If Rules are not specified, DefaultBackend must be specified.
	// If DefaultBackend is not set, the handling of requests that do not match any
	// of the rules will be up to the Ingress controller.
	// +optional
	DefaultBackend *IngressBackend

	// tls represents the TLS configuration. Currently the ingress only supports a
	// single TLS port, 443. If multiple members of this list specify different hosts,
	// they will be multiplexed on the same port according to the hostname specified
	// through the SNI TLS extension, if the ingress controller fulfilling the
	// ingress supports SNI.
	// +listType=atomic
	// +optional
	TLS []IngressTLS

	// rules is a list of host rules used to configure the Ingress. If unspecified, or
	// no rule matches, all traffic is sent to the default backend.
	// +listType=atomic
	// +optional
	Rules []IngressRule
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IngressClass represents the class of the Ingress, referenced by the Ingress
// Spec. The `ingressclass.kubernetes.io/is-default-class` annotation can be
// used to indicate that an IngressClass should be considered default. When a
// single IngressClass resource has this annotation set to true, new Ingress
// resources without a class specified will be assigned this default class.
type IngressClass struct {
	metav1.TypeMeta

	metav1.ObjectMeta

	// spec is the desired state of the IngressClass.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec IngressClassSpec
}

// IngressClassSpec provides information about the class of an Ingress.
type IngressClassSpec struct {
	// controller refers to the name of the controller that should handle this
	// class. This allows for different "flavors" that are controlled by the
	// same controller. For example, you may have different parameters for the
	// same implementing controller. This should be specified as a
	// domain-prefixed path no more than 250 characters in length, e.g.
	// "acme.io/ingress-controller". This field is immutable.
	Controller string

	// parameters is a link to a custom resource containing additional
	// configuration for the controller. This is optional if the controller does
	// not require extra parameters.
	// +optional
	Parameters *IngressClassParametersReference
}

const (
	// IngressClassParametersReferenceScopeNamespace indicates that the
	// referenced Parameters resource is namespace-scoped.
	IngressClassParametersReferenceScopeNamespace = "Namespace"
	// IngressClassParametersReferenceScopeCluster indicates that the
	// referenced Parameters resource is cluster-scoped.
	IngressClassParametersReferenceScopeCluster = "Cluster"
)

// IngressClassParametersReference identifies an API object. This can be used
// to specify a cluster or namespace-scoped resource.
type IngressClassParametersReference struct {
	// apiGroup is the group for the resource being referenced. If apiGroup is
	// not specified, the specified kind must be in the core API group. For any
	// other third-party types, apiGroup is required.
	// +optional
	APIGroup *string

	// kind is the type of resource being referenced.
	Kind string

	// name is the name of resource being referenced.
	Name string

	// scope represents if this refers to a cluster or namespace scoped resource.
	// This may be set to "Cluster" (default) or "Namespace".
	// +optional
	Scope *string

	// namespace is the namespace of the resource being referenced. This field is
	// required when scope is set to "Namespace" and must be unset when scope is set to
	// "Cluster".
	// +optional
	Namespace *string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IngressClassList is a collection of IngressClasses.
type IngressClassList struct {
	metav1.TypeMeta

	// Standard object's metadata.
	// +optional
	metav1.ListMeta

	// items is the list of IngressClasses.
	Items []IngressClass
}

// IngressTLS describes the transport layer security associated with an ingress.
type IngressTLS struct {
	// hosts is a list of hosts included in the TLS certificate. The values in
	// this list must match the name/s used in the tlsSecret. Defaults to the
	// wildcard host setting for the loadbalancer controller fulfilling this
	// Ingress, if left unspecified.
	// +listType=atomic
	// +optional
	Hosts []string

	// secretName is the name of the secret used to terminate TLS traffic on
	// port 443. Field is left optional to allow TLS routing based on SNI
	// hostname alone. If the SNI host in a listener conflicts with the "Host"
	// header field used by an IngressRule, the SNI host is used for termination
	// and value of the "Host" header is used for routing.
	// +optional
	SecretName string
	// TODO: Consider specifying different modes of termination, protocols etc.
}

// IngressStatus describes the current state of the Ingress.
type IngressStatus struct {
	// loadBalancer contains the current status of the load-balancer.
	// +optional
	LoadBalancer IngressLoadBalancerStatus
}

// IngressLoadBalancerStatus represents the status of a load-balancer
type IngressLoadBalancerStatus struct {
	// ingress is a list containing ingress points for the load-balancer.
	// +optional
	Ingress []IngressLoadBalancerIngress
}

// IngressLoadBalancerIngress represents the status of a load-balancer ingress point.
type IngressLoadBalancerIngress struct {
	// ip is set for load-balancer ingress points that are IP based.
	// +optional
	IP string

	// hostname is set for load-balancer ingress points that are DNS based.
	// +optional
	Hostname string

	// ports provides information about the ports exposed by this LoadBalancer.
	// +optional
	Ports []IngressPortStatus
}

// IngressPortStatus represents the error condition of an ingress port
type IngressPortStatus struct {
	// port is the port number of the ingress port.
	Port int32

	// protocol is the protocol of the ingress port.
	Protocol api.Protocol

	// error indicates a problem on this port.
	// The format of the error must comply with the following rules:
	// - Kubernetes-defined error values use CamelCase names
	// - Provider-specific error values must follow label-name style (e.g.
	//   example.com/name).
	Error *string
}

// IngressRule represents the rules mapping the paths under a specified host to
// the related backend services. Incoming requests are first evaluated for a
// host match, then routed to the backend associated with the matching
// IngressRuleValue.
type IngressRule struct {
	// host is the fully qualified domain name of a network host, as defined by RFC 3986.
	// Note the following deviations from the "host" part of the
	// URI as defined in RFC 3986:
	// 1. IPs are not allowed. Currently an IngressRuleValue can only apply to
	//    the IP in the Spec of the parent Ingress.
	// 2. The `:` delimiter is not respected because ports are not allowed.
	//	  Currently the port of an Ingress is implicitly :80 for http and
	//	  :443 for https.
	// Both these may change in the future.
	// Incoming requests are matched against the host before the
	// IngressRuleValue. If the host is unspecified, the Ingress routes all
	// traffic based on the specified IngressRuleValue.
	//
	// host can be "precise" which is a domain name without the terminating dot of
	// a network host (e.g. "foo.bar.com") or "wildcard", which is a domain name
	// prefixed with a single wildcard label (e.g. "*.foo.com").
	// The wildcard character '*' must appear by itself as the first DNS label and
	// matches only a single label. You cannot have a wildcard label by itself (e.g. Host == "*").
	// Requests will be matched against the host field in the following way:
	// 1. If host is precise, the request matches this rule if the http host header is equal to Host.
	// 2. If host is a wildcard, then the request matches this rule if the http host header
	// is to equal to the suffix (removing the first label) of the wildcard rule.
	// +optional
	Host string

	// IngressRuleValue represents a rule to route requests for this
	// IngressRule. If unspecified, the rule defaults to a http catch-all.
	// Whether that sends just traffic matching the host to the default backend
	// or all traffic to the default backend, is left to the controller
	// fulfilling the Ingress. Http is currently the only supported
	// IngressRuleValue.
	// +optional
	IngressRuleValue
}

// IngressRuleValue represents a rule to apply against incoming requests. If the
// rule is satisfied, the request is routed to the specified backend. Currently
// mixing different types of rules in a single Ingress is disallowed, so exactly
// one of the following must be set.
type IngressRuleValue struct {
	//TODO:
	// 1. Consider renaming this resource and the associated rules so they
	// aren't tied to Ingress. They can be used to route intra-cluster traffic.
	// 2. Consider adding fields for ingress-type specific global options
	// usable by a loadbalancer, like http keep-alive.

	// +optional
	HTTP *HTTPIngressRuleValue
}

// HTTPIngressRuleValue is a list of http selectors pointing to backends.
// In the example: http://<host>/<path>?<searchpart> -> backend where
// where parts of the url correspond to RFC 3986, this resource will be used
// to match against everything after the last '/' and before the first '?'
// or '#'.
type HTTPIngressRuleValue struct {
	// paths is a collection of paths that map requests to backends.
	// +listType=atomic
	Paths []HTTPIngressPath
	// TODO: Consider adding fields for ingress-type specific global
	// options usable by a loadbalancer, like http keep-alive.
}

// PathType represents the type of path referred to by a HTTPIngressPath.
type PathType string

const (
	// PathTypeExact matches the URL path exactly and with case sensitivity.
	PathTypeExact = PathType("Exact")

	// PathTypePrefix matches based on a URL path prefix split by '/'. Matching
	// is case sensitive and done on a path element by element basis. A path
	// element refers to the list of labels in the path split by the '/'
	// separator. A request is a match for path p if every p is an element-wise
	// prefix of p of the request path. Note that if the last element of the
	// path is a substring of the last element in request path, it is not a
	// match (e.g. /foo/bar matches /foo/bar/baz, but does not match
	// /foo/barbaz). If multiple matching paths exist in an Ingress spec, the
	// longest matching path is given priority.
	// Examples:
	// - /foo/bar does not match requests to /foo/barbaz
	// - /foo/bar matches request to /foo/bar and /foo/bar/baz
	// - /foo and /foo/ both match requests to /foo and /foo/. If both paths are
	//   present in an Ingress spec, the longest matching path (/foo/) is given
	//   priority.
	PathTypePrefix = PathType("Prefix")

	// PathTypeImplementationSpecific matching is up to the IngressClass.
	// Implementations can treat this as a separate PathType or treat it
	// identically to Prefix or Exact path types.
	PathTypeImplementationSpecific = PathType("ImplementationSpecific")
)

// HTTPIngressPath associates a path with a backend. Incoming urls matching the
// path are forwarded to the backend.
type HTTPIngressPath struct {
	// path is matched against the path of an incoming request. Currently it can
	// contain characters disallowed from the conventional "path" part of a URL
	// as defined by RFC 3986. Paths must begin with a '/' and must be present
	// when using PathType with value "Exact" or "Prefix".
	// +optional
	Path string

	// pathType determines the interpretation of the path matching. PathType can
	// be one of Exact, Prefix, or ImplementationSpecific. Implementations are
	// required to support all path types.
	// +optional
	PathType *PathType

	// backend defines the referenced service endpoint to which the traffic
	// will be forwarded to.
	Backend IngressBackend
}

// IngressBackend describes all endpoints for a given service and port.
type IngressBackend struct {
	// service references a service as a backend.
	// This is a mutually exclusive setting with "Resource".
	// +optional
	Service *IngressServiceBackend

	// resource is an ObjectRef to another Kubernetes resource in the namespace
	// of the Ingress object. If resource is specified, a service.Name and
	// service.Port must not be specified.
	// This is a mutually exclusive setting with "Service".
	// +optional
	Resource *api.TypedLocalObjectReference
}

// IngressServiceBackend references a Kubernetes Service as a Backend.
type IngressServiceBackend struct {
	// name is the referenced service. The service must exist in
	// the same namespace as the Ingress object.
	Name string

	// port of the referenced service. A port name or port number
	// is required for a IngressServiceBackend.
	Port ServiceBackendPort
}

// ServiceBackendPort is the service port being referenced.
type ServiceBackendPort struct {
	// name is the name of the port on the Service.
	// This must be an IANA_SVC_NAME (following RFC6335).
	// This is a mutually exclusive setting with "Number".
	// +optional
	Name string

	// number is the numerical port number (e.g. 80) on the Service.
	// This is a mutually exclusive setting with "Name".
	// +optional
	Number int32
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IPAddress represents a single IP of a single IP Family. The object is designed to be used by APIs
// that operate on IP addresses. The object is used by the Service core API for allocation of IP addresses.
// An IP address can be represented in different formats, to guarantee the uniqueness of the IP,
// the name of the object is the IP address in canonical format, four decimal digits separated
// by dots suppressing leading zeros for IPv4 and the representation defined by RFC 5952 for IPv6.
// Valid: 192.168.1.5 or 2001:db8::1 or 2001:db8:aaaa:bbbb:cccc:dddd:eeee:1
// Invalid: 10.01.2.3 or 2001:db8:0:0:0::1
type IPAddress struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta
	// +optional
	Spec IPAddressSpec
}

// IPAddressSpec describe the attributes in an IP Address,
type IPAddressSpec struct {
	// ParentRef references the resource that an IPAddress is attached to.
	// An IPAddress must reference a parent object.
	// +required
	ParentRef *ParentReference
}
type ParentReference struct {
	// Group is the group of the object being referenced.
	Group string
	// Resource is the resource of the object being referenced.
	Resource string
	// Namespace is the namespace of the object being referenced.
	Namespace string
	// Name is the name of the object being referenced.
	Name string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IPAddressList contains a list of IPAddress.
type IPAddressList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// Items is the list of IPAddress
	Items []IPAddress
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceCIDR defines a range of IP addresses using CIDR format (e.g. 192.168.0.0/24 or 2001:db2::/64).
// This range is used to allocate ClusterIPs to Service objects.
type ServiceCIDR struct {
	metav1.TypeMeta
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta
	// spec is the desired state of the ServiceCIDR.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec ServiceCIDRSpec
	// status represents the current state of the ServiceCIDR.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status ServiceCIDRStatus
}

type ServiceCIDRSpec struct {
	// CIDRs defines the IP blocks in CIDR notation (e.g. "192.168.0.0/24" or "2001:db8::/64")
	// from which to assign service cluster IPs. Max of two CIDRs is allowed, one of each IP family.
	// This field is immutable.
	// +optional
	CIDRs []string
}

// ServiceCIDRStatus describes the current state of the ServiceCIDR.
type ServiceCIDRStatus struct {
	// conditions holds an array of metav1.Condition that describe the state of the ServiceCIDR.
	Conditions []metav1.Condition
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.27

// ServiceCIDRList contains a list of ServiceCIDR objects.
type ServiceCIDRList struct {
	metav1.TypeMeta
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta
	// items is the list of ServiceCIDRs.
	Items []ServiceCIDR
}
