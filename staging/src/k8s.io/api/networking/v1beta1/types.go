/*
Copyright 2019 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/intstr"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.14
// +k8s:prerelease-lifecycle-gen:deprecated=1.19
// +k8s:prerelease-lifecycle-gen:replacement=networking.k8s.io,v1,Ingress

// Ingress is a collection of rules that allow inbound connections to reach the
// endpoints defined by a backend. An Ingress can be configured to give services
// externally-reachable urls, load balance traffic, terminate SSL, offer name
// based virtual hosting etc.
type Ingress struct {
	metav1.TypeMeta `json:",inline"`

	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec is the desired state of the Ingress.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec IngressSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// status is the current state of the Ingress.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status IngressStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.14
// +k8s:prerelease-lifecycle-gen:deprecated=1.19
// +k8s:prerelease-lifecycle-gen:replacement=networking.k8s.io,v1,IngressList

// IngressList is a collection of Ingress.
type IngressList struct {
	metav1.TypeMeta `json:",inline"`

	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of Ingress.
	Items []Ingress `json:"items" protobuf:"bytes,2,rep,name=items"`
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
	IngressClassName *string `json:"ingressClassName,omitempty" protobuf:"bytes,4,opt,name=ingressClassName"`

	// backend is the default backend capable of servicing requests that don't match any
	// rule. At least one of 'backend' or 'rules' must be specified. This field
	// is optional to allow the loadbalancer controller or defaulting logic to
	// specify a global default.
	// +optional
	Backend *IngressBackend `json:"backend,omitempty" protobuf:"bytes,1,opt,name=backend"`

	// tls represents the TLS configuration. Currently the Ingress only supports a
	// single TLS port, 443. If multiple members of this list specify different hosts,
	// they will be multiplexed on the same port according to the hostname specified
	// through the SNI TLS extension, if the ingress controller fulfilling the
	// ingress supports SNI.
	// +optional
	// +listType=atomic
	TLS []IngressTLS `json:"tls,omitempty" protobuf:"bytes,2,rep,name=tls"`

	// rules is a list of host rules used to configure the Ingress. If unspecified, or
	// no rule matches, all traffic is sent to the default backend.
	// +optional
	// +listType=atomic
	Rules []IngressRule `json:"rules,omitempty" protobuf:"bytes,3,rep,name=rules"`
	// TODO: Add the ability to specify load-balancer IP through claims
}

// IngressTLS describes the transport layer security associated with an Ingress.
type IngressTLS struct {
	// hosts is a list of hosts included in the TLS certificate. The values in
	// this list must match the name/s used in the tlsSecret. Defaults to the
	// wildcard host setting for the loadbalancer controller fulfilling this
	// Ingress, if left unspecified.
	// +optional
	// +listType=atomic
	Hosts []string `json:"hosts,omitempty" protobuf:"bytes,1,rep,name=hosts"`

	// secretName is the name of the secret used to terminate TLS traffic on
	// port 443. Field is left optional to allow TLS routing based on SNI
	// hostname alone. If the SNI host in a listener conflicts with the "Host"
	// header field used by an IngressRule, the SNI host is used for termination
	// and value of the Host header is used for routing.
	// +optional
	SecretName string `json:"secretName,omitempty" protobuf:"bytes,2,opt,name=secretName"`
	// TODO: Consider specifying different modes of termination, protocols etc.
}

// IngressStatus describes the current state of the Ingress.
type IngressStatus struct {
	// loadBalancer contains the current status of the load-balancer.
	// +optional
	LoadBalancer IngressLoadBalancerStatus `json:"loadBalancer,omitempty" protobuf:"bytes,1,opt,name=loadBalancer"`
}

// LoadBalancerStatus represents the status of a load-balancer.
type IngressLoadBalancerStatus struct {
	// ingress is a list containing ingress points for the load-balancer.
	// +optional
	// +listType=atomic
	Ingress []IngressLoadBalancerIngress `json:"ingress,omitempty" protobuf:"bytes,1,rep,name=ingress"`
}

// IngressLoadBalancerIngress represents the status of a load-balancer ingress point.
type IngressLoadBalancerIngress struct {
	// ip is set for load-balancer ingress points that are IP based.
	// +optional
	IP string `json:"ip,omitempty" protobuf:"bytes,1,opt,name=ip"`

	// hostname is set for load-balancer ingress points that are DNS based.
	// +optional
	Hostname string `json:"hostname,omitempty" protobuf:"bytes,2,opt,name=hostname"`

	// ports provides information about the ports exposed by this LoadBalancer.
	// +listType=atomic
	// +optional
	Ports []IngressPortStatus `json:"ports,omitempty" protobuf:"bytes,4,rep,name=ports"`
}

// IngressPortStatus represents the error condition of a service port
type IngressPortStatus struct {
	// port is the port number of the ingress port.
	Port int32 `json:"port" protobuf:"varint,1,opt,name=port"`

	// protocol is the protocol of the ingress port.
	// The supported values are: "TCP", "UDP", "SCTP"
	Protocol v1.Protocol `json:"protocol" protobuf:"bytes,2,opt,name=protocol,casttype=Protocol"`

	// error is to record the problem with the service port
	// The format of the error shall comply with the following rules:
	// - built-in error values shall be specified in this file and those shall use
	//   CamelCase names
	// - cloud provider specific error values must have names that comply with the
	//   format foo.example.com/CamelCase.
	// ---
	// The regex it matches is (dns1123SubdomainFmt/)?(qualifiedNameFmt)
	// +optional
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^([a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*/)?(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])$`
	// +kubebuilder:validation:MaxLength=316
	Error *string `json:"error,omitempty" protobuf:"bytes,3,opt,name=error"`
}

// IngressRule represents the rules mapping the paths under a specified host to
// the related backend services. Incoming requests are first evaluated for a host
// match, then routed to the backend associated with the matching IngressRuleValue.
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
	// Requests will be matched against the Host field in the following way:
	// 1. If Host is precise, the request matches this rule if the http host header is equal to Host.
	// 2. If Host is a wildcard, then the request matches this rule if the http host header
	// is to equal to the suffix (removing the first label) of the wildcard rule.
	// +optional
	Host string `json:"host,omitempty" protobuf:"bytes,1,opt,name=host"`

	// IngressRuleValue represents a rule to route requests for this IngressRule.
	// If unspecified, the rule defaults to a http catch-all. Whether that sends
	// just traffic matching the host to the default backend or all traffic to the
	// default backend, is left to the controller fulfilling the Ingress. Http is
	// currently the only supported IngressRuleValue.
	// +optional
	IngressRuleValue `json:",inline,omitempty" protobuf:"bytes,2,opt,name=ingressRuleValue"`
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
	HTTP *HTTPIngressRuleValue `json:"http,omitempty" protobuf:"bytes,1,opt,name=http"`
}

// HTTPIngressRuleValue is a list of http selectors pointing to backends.
// In the example: http://<host>/<path>?<searchpart> -> backend where
// where parts of the url correspond to RFC 3986, this resource will be used
// to match against everything after the last '/' and before the first '?'
// or '#'.
type HTTPIngressRuleValue struct {
	// paths is a collection of paths that map requests to backends.
	// +listType=atomic
	Paths []HTTPIngressPath `json:"paths" protobuf:"bytes,1,rep,name=paths"`
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
	Path string `json:"path,omitempty" protobuf:"bytes,1,opt,name=path"`

	// pathType determines the interpretation of the path matching. PathType can
	// be one of the following values:
	// * Exact: Matches the URL path exactly.
	// * Prefix: Matches based on a URL path prefix split by '/'. Matching is
	//   done on a path element by element basis. A path element refers is the
	//   list of labels in the path split by the '/' separator. A request is a
	//   match for path p if every p is an element-wise prefix of p of the
	//   request path. Note that if the last element of the path is a substring
	//   of the last element in request path, it is not a match (e.g. /foo/bar
	//   matches /foo/bar/baz, but does not match /foo/barbaz).
	// * ImplementationSpecific: Interpretation of the Path matching is up to
	//   the IngressClass. Implementations can treat this as a separate PathType
	//   or treat it identically to Prefix or Exact path types.
	// Implementations are required to support all path types.
	// Defaults to ImplementationSpecific.
	PathType *PathType `json:"pathType,omitempty" protobuf:"bytes,3,opt,name=pathType"`

	// backend defines the referenced service endpoint to which the traffic
	// will be forwarded to.
	Backend IngressBackend `json:"backend" protobuf:"bytes,2,opt,name=backend"`
}

// IngressBackend describes all endpoints for a given service and port.
type IngressBackend struct {
	// serviceName specifies the name of the referenced service.
	// +optional
	ServiceName string `json:"serviceName,omitempty" protobuf:"bytes,1,opt,name=serviceName"`

	// servicePort Specifies the port of the referenced service.
	// +optional
	ServicePort intstr.IntOrString `json:"servicePort,omitempty" protobuf:"bytes,2,opt,name=servicePort"`

	// resource is an ObjectRef to another Kubernetes resource in the namespace
	// of the Ingress object. If resource is specified, serviceName and servicePort
	// must not be specified.
	// +optional
	Resource *v1.TypedLocalObjectReference `json:"resource,omitempty" protobuf:"bytes,3,opt,name=resource"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.18
// +k8s:prerelease-lifecycle-gen:deprecated=1.19
// +k8s:prerelease-lifecycle-gen:replacement=networking.k8s.io,v1,IngressClassList

// IngressClass represents the class of the Ingress, referenced by the Ingress
// Spec. The `ingressclass.kubernetes.io/is-default-class` annotation can be
// used to indicate that an IngressClass should be considered default. When a
// single IngressClass resource has this annotation set to true, new Ingress
// resources without a class specified will be assigned this default class.
type IngressClass struct {
	metav1.TypeMeta `json:",inline"`

	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec is the desired state of the IngressClass.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec IngressClassSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// IngressClassSpec provides information about the class of an Ingress.
type IngressClassSpec struct {
	// controller refers to the name of the controller that should handle this
	// class. This allows for different "flavors" that are controlled by the
	// same controller. For example, you may have different parameters for the
	// same implementing controller. This should be specified as a
	// domain-prefixed path no more than 250 characters in length, e.g.
	// "acme.io/ingress-controller". This field is immutable.
	Controller string `json:"controller,omitempty" protobuf:"bytes,1,opt,name=controller"`

	// parameters is a link to a custom resource containing additional
	// configuration for the controller. This is optional if the controller does
	// not require extra parameters.
	// +optional
	Parameters *IngressClassParametersReference `json:"parameters,omitempty" protobuf:"bytes,2,opt,name=parameters"`
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
	// apiGroup is the group for the resource being referenced. If APIGroup is
	// not specified, the specified Kind must be in the core API group. For any
	// other third-party types, APIGroup is required.
	// +optional
	APIGroup *string `json:"apiGroup,omitempty" protobuf:"bytes,1,opt,name=aPIGroup"`

	// kind is the type of resource being referenced.
	Kind string `json:"kind" protobuf:"bytes,2,opt,name=kind"`

	// name is the name of resource being referenced.
	Name string `json:"name" protobuf:"bytes,3,opt,name=name"`

	// scope represents if this refers to a cluster or namespace scoped resource.
	// This may be set to "Cluster" (default) or "Namespace".
	Scope *string `json:"scope" protobuf:"bytes,4,opt,name=scope"`

	// namespace is the namespace of the resource being referenced. This field is
	// required when scope is set to "Namespace" and must be unset when scope is set to
	// "Cluster".
	// +optional
	Namespace *string `json:"namespace,omitempty" protobuf:"bytes,5,opt,name=namespace"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.18
// +k8s:prerelease-lifecycle-gen:deprecated=1.19
// +k8s:prerelease-lifecycle-gen:replacement=networking.k8s.io,v1,IngressClassList

// IngressClassList is a collection of IngressClasses.
type IngressClassList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of IngressClasses.
	Items []IngressClass `json:"items" protobuf:"bytes,2,rep,name=items"`
}
