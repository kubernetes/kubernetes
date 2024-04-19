package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Ingress holds cluster-wide information about ingress, including the default ingress domain
// used for routes. The canonical name is `cluster`.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/470
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=ingresses,scope=Cluster
// +kubebuilder:subresource:status
type Ingress struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec IngressSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status IngressStatus `json:"status"`
}

type IngressSpec struct {
	// domain is used to generate a default host name for a route when the
	// route's host name is empty. The generated host name will follow this
	// pattern: "<route-name>.<route-namespace>.<domain>".
	//
	// It is also used as the default wildcard domain suffix for ingress. The
	// default ingresscontroller domain will follow this pattern: "*.<domain>".
	//
	// Once set, changing domain is not currently supported.
	Domain string `json:"domain"`

	// appsDomain is an optional domain to use instead of the one specified
	// in the domain field when a Route is created without specifying an explicit
	// host. If appsDomain is nonempty, this value is used to generate default
	// host values for Route. Unlike domain, appsDomain may be modified after
	// installation.
	// This assumes a new ingresscontroller has been setup with a wildcard
	// certificate.
	// +optional
	AppsDomain string `json:"appsDomain,omitempty"`

	// componentRoutes is an optional list of routes that are managed by OpenShift components
	// that a cluster-admin is able to configure the hostname and serving certificate for.
	// The namespace and name of each route in this list should match an existing entry in the
	// status.componentRoutes list.
	//
	// To determine the set of configurable Routes, look at namespace and name of entries in the
	// .status.componentRoutes list, where participating operators write the status of
	// configurable routes.
	// +optional
	// +listType=map
	// +listMapKey=namespace
	// +listMapKey=name
	ComponentRoutes []ComponentRouteSpec `json:"componentRoutes,omitempty"`

	// requiredHSTSPolicies specifies HSTS policies that are required to be set on newly created  or updated routes
	// matching the domainPattern/s and namespaceSelector/s that are specified in the policy.
	// Each requiredHSTSPolicy must have at least a domainPattern and a maxAge to validate a route HSTS Policy route
	// annotation, and affect route admission.
	//
	// A candidate route is checked for HSTS Policies if it has the HSTS Policy route annotation:
	// "haproxy.router.openshift.io/hsts_header"
	// E.g. haproxy.router.openshift.io/hsts_header: max-age=31536000;preload;includeSubDomains
	//
	// - For each candidate route, if it matches a requiredHSTSPolicy domainPattern and optional namespaceSelector,
	// then the maxAge, preloadPolicy, and includeSubdomainsPolicy must be valid to be admitted.  Otherwise, the route
	// is rejected.
	// - The first match, by domainPattern and optional namespaceSelector, in the ordering of the RequiredHSTSPolicies
	// determines the route's admission status.
	// - If the candidate route doesn't match any requiredHSTSPolicy domainPattern and optional namespaceSelector,
	// then it may use any HSTS Policy annotation.
	//
	// The HSTS policy configuration may be changed after routes have already been created. An update to a previously
	// admitted route may then fail if the updated route does not conform to the updated HSTS policy configuration.
	// However, changing the HSTS policy configuration will not cause a route that is already admitted to stop working.
	//
	// Note that if there are no RequiredHSTSPolicies, any HSTS Policy annotation on the route is valid.
	// +optional
	RequiredHSTSPolicies []RequiredHSTSPolicy `json:"requiredHSTSPolicies,omitempty"`

	// loadBalancer contains the load balancer details in general which are not only specific to the underlying infrastructure
	// provider of the current cluster and are required for Ingress Controller to work on OpenShift.
	// +optional
	LoadBalancer LoadBalancer `json:"loadBalancer,omitempty"`
}

// IngressPlatformSpec holds the desired state of Ingress specific to the underlying infrastructure provider
// of the current cluster. Since these are used at spec-level for the underlying cluster, it
// is supposed that only one of the spec structs is set.
// +union
type IngressPlatformSpec struct {
	// type is the underlying infrastructure provider for the cluster.
	// Allowed values are "AWS", "Azure", "BareMetal", "GCP", "Libvirt",
	// "OpenStack", "VSphere", "oVirt", "KubeVirt", "EquinixMetal", "PowerVS",
	// "AlibabaCloud", "Nutanix" and "None". Individual components may not support all platforms,
	// and must handle unrecognized platforms as None if they do not support that platform.
	//
	// +unionDiscriminator
	Type PlatformType `json:"type"`

	// aws contains settings specific to the Amazon Web Services infrastructure provider.
	// +optional
	AWS *AWSIngressSpec `json:"aws,omitempty"`
}

type LoadBalancer struct {
	// platform holds configuration specific to the underlying
	// infrastructure provider for the ingress load balancers.
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// +optional
	Platform IngressPlatformSpec `json:"platform,omitempty"`
}

// AWSIngressSpec holds the desired state of the Ingress for Amazon Web Services infrastructure provider.
// This only includes fields that can be modified in the cluster.
// +union
type AWSIngressSpec struct {
	// type allows user to set a load balancer type.
	// When this field is set the default ingresscontroller will get created using the specified LBType.
	// If this field is not set then the default ingress controller of LBType Classic will be created.
	// Valid values are:
	//
	// * "Classic": A Classic Load Balancer that makes routing decisions at either
	//   the transport layer (TCP/SSL) or the application layer (HTTP/HTTPS). See
	//   the following for additional details:
	//
	//     https://docs.aws.amazon.com/AmazonECS/latest/developerguide/load-balancer-types.html#clb
	//
	// * "NLB": A Network Load Balancer that makes routing decisions at the
	//   transport layer (TCP/SSL). See the following for additional details:
	//
	//     https://docs.aws.amazon.com/AmazonECS/latest/developerguide/load-balancer-types.html#nlb
	// +unionDiscriminator
	// +kubebuilder:validation:Enum:=NLB;Classic
	// +kubebuilder:validation:Required
	Type AWSLBType `json:"type,omitempty"`
}

type AWSLBType string

const (
	// NLB is the Network Load Balancer Type of AWS. Using NLB one can set NLB load balancer type for the default ingress controller.
	NLB AWSLBType = "NLB"

	// Classic is the Classic Load Balancer Type of AWS. Using CLassic one can set Classic load balancer type for the default ingress controller.
	Classic AWSLBType = "Classic"
)

// ConsumingUser is an alias for string which we add validation to. Currently only service accounts are supported.
// +kubebuilder:validation:Pattern="^system:serviceaccount:[a-z0-9]([-a-z0-9]*[a-z0-9])?:[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=512
type ConsumingUser string

// Hostname is a host name as defined by RFC-1123.
// + ---
// + The left operand of the | is the original kubebuilder hostname validation format, which is incorrect because it
// + allows upper case letters, disallows hyphen or number in the TLD, and allows labels to start/end in non-alphanumeric
// + characters.  See https://bugzilla.redhat.com/show_bug.cgi?id=2039256.
// + ^([a-zA-Z0-9\p{S}\p{L}]((-?[a-zA-Z0-9\p{S}\p{L}]{0,62})?)|([a-zA-Z0-9\p{S}\p{L}](([a-zA-Z0-9-\p{S}\p{L}]{0,61}[a-zA-Z0-9\p{S}\p{L}])?)(\.)){1,}([a-zA-Z\p{L}]){2,63})$
// +
// + The right operand of the | is a new pattern that mimics the current API route admission validation on hostname,
// + except that it allows hostnames longer than the maximum length:
// + ^(([a-z0-9][-a-z0-9]{0,61}[a-z0-9]|[a-z0-9]{1,63})[\.]){0,}([a-z0-9][-a-z0-9]{0,61}[a-z0-9]|[a-z0-9]{1,63})$
// +
// + Both operand patterns are made available so that modifications on ingress spec can still happen after an invalid hostname
// + was saved via validation by the incorrect left operand of the | operator.
// +
// +kubebuilder:validation:Pattern=`^([a-zA-Z0-9\p{S}\p{L}]((-?[a-zA-Z0-9\p{S}\p{L}]{0,62})?)|([a-zA-Z0-9\p{S}\p{L}](([a-zA-Z0-9-\p{S}\p{L}]{0,61}[a-zA-Z0-9\p{S}\p{L}])?)(\.)){1,}([a-zA-Z\p{L}]){2,63})$|^(([a-z0-9][-a-z0-9]{0,61}[a-z0-9]|[a-z0-9]{1,63})[\.]){0,}([a-z0-9][-a-z0-9]{0,61}[a-z0-9]|[a-z0-9]{1,63})$`
type Hostname string

type IngressStatus struct {
	// componentRoutes is where participating operators place the current route status for routes whose
	// hostnames and serving certificates can be customized by the cluster-admin.
	// +optional
	// +listType=map
	// +listMapKey=namespace
	// +listMapKey=name
	ComponentRoutes []ComponentRouteStatus `json:"componentRoutes,omitempty"`

	// defaultPlacement is set at installation time to control which
	// nodes will host the ingress router pods by default. The options are
	// control-plane nodes or worker nodes.
	//
	// This field works by dictating how the Cluster Ingress Operator will
	// consider unset replicas and nodePlacement fields in IngressController
	// resources when creating the corresponding Deployments.
	//
	// See the documentation for the IngressController replicas and nodePlacement
	// fields for more information.
	//
	// When omitted, the default value is Workers
	//
	// +kubebuilder:validation:Enum:="ControlPlane";"Workers";""
	// +optional
	DefaultPlacement DefaultPlacement `json:"defaultPlacement"`
}

// ComponentRouteSpec allows for configuration of a route's hostname and serving certificate.
type ComponentRouteSpec struct {
	// namespace is the namespace of the route to customize.
	//
	// The namespace and name of this componentRoute must match a corresponding
	// entry in the list of status.componentRoutes if the route is to be customized.
	// +kubebuilder:validation:Pattern=^[a-z0-9]([-a-z0-9]*[a-z0-9])?$
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Required
	// +required
	Namespace string `json:"namespace"`

	// name is the logical name of the route to customize.
	//
	// The namespace and name of this componentRoute must match a corresponding
	// entry in the list of status.componentRoutes if the route is to be customized.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:Required
	// +required
	Name string `json:"name"`

	// hostname is the hostname that should be used by the route.
	// +kubebuilder:validation:Required
	// +required
	Hostname Hostname `json:"hostname"`

	// servingCertKeyPairSecret is a reference to a secret of type `kubernetes.io/tls` in the openshift-config namespace.
	// The serving cert/key pair must match and will be used by the operator to fulfill the intent of serving with this name.
	// If the custom hostname uses the default routing suffix of the cluster,
	// the Secret specification for a serving certificate will not be needed.
	// +optional
	ServingCertKeyPairSecret SecretNameReference `json:"servingCertKeyPairSecret"`
}

// ComponentRouteStatus contains information allowing configuration of a route's hostname and serving certificate.
type ComponentRouteStatus struct {
	// namespace is the namespace of the route to customize. It must be a real namespace. Using an actual namespace
	// ensures that no two components will conflict and the same component can be installed multiple times.
	//
	// The namespace and name of this componentRoute must match a corresponding
	// entry in the list of spec.componentRoutes if the route is to be customized.
	// +kubebuilder:validation:Pattern=^[a-z0-9]([-a-z0-9]*[a-z0-9])?$
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Required
	// +required
	Namespace string `json:"namespace"`

	// name is the logical name of the route to customize. It does not have to be the actual name of a route resource
	// but it cannot be renamed.
	//
	// The namespace and name of this componentRoute must match a corresponding
	// entry in the list of spec.componentRoutes if the route is to be customized.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:Required
	// +required
	Name string `json:"name"`

	// defaultHostname is the hostname of this route prior to customization.
	// +kubebuilder:validation:Required
	// +required
	DefaultHostname Hostname `json:"defaultHostname"`

	// consumingUsers is a slice of ServiceAccounts that need to have read permission on the servingCertKeyPairSecret secret.
	// +kubebuilder:validation:MaxItems=5
	// +optional
	ConsumingUsers []ConsumingUser `json:"consumingUsers,omitempty"`

	// currentHostnames is the list of current names used by the route. Typically, this list should consist of a single
	// hostname, but if multiple hostnames are supported by the route the operator may write multiple entries to this list.
	// +kubebuilder:validation:MinItems=1
	// +optional
	CurrentHostnames []Hostname `json:"currentHostnames,omitempty"`

	// conditions are used to communicate the state of the componentRoutes entry.
	//
	// Supported conditions include Available, Degraded and Progressing.
	//
	// If available is true, the content served by the route can be accessed by users. This includes cases
	// where a default may continue to serve content while the customized route specified by the cluster-admin
	// is being configured.
	//
	// If Degraded is true, that means something has gone wrong trying to handle the componentRoutes entry.
	// The currentHostnames field may or may not be in effect.
	//
	// If Progressing is true, that means the component is taking some action related to the componentRoutes entry.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// relatedObjects is a list of resources which are useful when debugging or inspecting how spec.componentRoutes is applied.
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:Required
	// +required
	RelatedObjects []ObjectReference `json:"relatedObjects"`
}

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +openshift:compatibility-gen:level=1
type IngressList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []Ingress `json:"items"`
}

// DefaultPlacement defines the default placement of ingress router pods.
type DefaultPlacement string

const (
	// "Workers" is for having router pods placed on worker nodes by default.
	DefaultPlacementWorkers DefaultPlacement = "Workers"

	// "ControlPlane" is for having router pods placed on control-plane nodes by default.
	DefaultPlacementControlPlane DefaultPlacement = "ControlPlane"
)
