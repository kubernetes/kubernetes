package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Ingress holds cluster-wide information about ingress, including the default ingress domain
// used for routes. The canonical name is `cluster`.
type Ingress struct {
	metav1.TypeMeta   `json:",inline"`
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
	ComponentRoutes []ComponentRouteSpec `json:"componentRoutes,omitempty"`
}

// ConsumingUser is an alias for string which we add validation to. Currently only service accounts are supported.
// +kubebuilder:validation:Pattern="^system:serviceaccount:[a-z0-9]([-a-z0-9]*[a-z0-9])?:[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=512
type ConsumingUser string

// Hostname is an alias for hostname string validation.
// +kubebuilder:validation:Format=hostname
type Hostname string

type IngressStatus struct {
	// componentRoutes is where participating operators place the current route status for routes whose
	// hostnames and serving certificates can be customized by the cluster-admin.
	// +optional
	ComponentRoutes []ComponentRouteStatus `json:"componentRoutes,omitempty"`
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
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// relatedObjects is a list of resources which are useful when debugging or inspecting how spec.componentRoutes is applied.
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:Required
	// +required
	RelatedObjects []ObjectReference `json:"relatedObjects"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type IngressList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Ingress `json:"items"`
}
