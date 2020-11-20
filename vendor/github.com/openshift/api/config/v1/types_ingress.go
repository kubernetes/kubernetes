package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

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
}

type IngressStatus struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type IngressList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Ingress `json:"items"`
}
