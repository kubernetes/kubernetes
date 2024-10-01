package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Proxy holds cluster-wide information on how to configure default proxies for the cluster. The canonical name is `cluster`
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/470
// +openshift:file-pattern=cvoRunLevel=0000_03,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=proxies,scope=Cluster
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations=release.openshift.io/bootstrap-required=true
type Proxy struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec holds user-settable values for the proxy configuration
	// +kubebuilder:validation:Required
	// +required
	Spec ProxySpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status ProxyStatus `json:"status"`
}

// ProxySpec contains cluster proxy creation configuration.
type ProxySpec struct {
	// httpProxy is the URL of the proxy for HTTP requests.  Empty means unset and will not result in an env var.
	// +optional
	HTTPProxy string `json:"httpProxy,omitempty"`

	// httpsProxy is the URL of the proxy for HTTPS requests.  Empty means unset and will not result in an env var.
	// +optional
	HTTPSProxy string `json:"httpsProxy,omitempty"`

	// noProxy is a comma-separated list of hostnames and/or CIDRs and/or IPs for which the proxy should not be used.
	// Empty means unset and will not result in an env var.
	// +optional
	NoProxy string `json:"noProxy,omitempty"`

	// readinessEndpoints is a list of endpoints used to verify readiness of the proxy.
	// +optional
	ReadinessEndpoints []string `json:"readinessEndpoints,omitempty"`

	// trustedCA is a reference to a ConfigMap containing a CA certificate bundle.
	// The trustedCA field should only be consumed by a proxy validator. The
	// validator is responsible for reading the certificate bundle from the required
	// key "ca-bundle.crt", merging it with the system default trust bundle,
	// and writing the merged trust bundle to a ConfigMap named "trusted-ca-bundle"
	// in the "openshift-config-managed" namespace. Clients that expect to make
	// proxy connections must use the trusted-ca-bundle for all HTTPS requests to
	// the proxy, and may use the trusted-ca-bundle for non-proxy HTTPS requests as
	// well.
	//
	// The namespace for the ConfigMap referenced by trustedCA is
	// "openshift-config". Here is an example ConfigMap (in yaml):
	//
	// apiVersion: v1
	// kind: ConfigMap
	// metadata:
	//  name: user-ca-bundle
	//  namespace: openshift-config
	//  data:
	//    ca-bundle.crt: |
	//      -----BEGIN CERTIFICATE-----
	//      Custom CA certificate bundle.
	//      -----END CERTIFICATE-----
	//
	// +optional
	TrustedCA ConfigMapNameReference `json:"trustedCA,omitempty"`
}

// ProxyStatus shows current known state of the cluster proxy.
type ProxyStatus struct {
	// httpProxy is the URL of the proxy for HTTP requests.
	// +optional
	HTTPProxy string `json:"httpProxy,omitempty"`

	// httpsProxy is the URL of the proxy for HTTPS requests.
	// +optional
	HTTPSProxy string `json:"httpsProxy,omitempty"`

	// noProxy is a comma-separated list of hostnames and/or CIDRs for which the proxy should not be used.
	// +optional
	NoProxy string `json:"noProxy,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ProxyList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []Proxy `json:"items"`
}
