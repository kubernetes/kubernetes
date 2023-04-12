package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Console holds cluster-wide configuration for the web console, including the
// logout URL, and reports the public URL of the console. The canonical name is
// `cluster`.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Console struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec ConsoleSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status ConsoleStatus `json:"status"`
}

// ConsoleSpec is the specification of the desired behavior of the Console.
type ConsoleSpec struct {
	// +optional
	Authentication ConsoleAuthentication `json:"authentication"`
}

// ConsoleStatus defines the observed status of the Console.
type ConsoleStatus struct {
	// The URL for the console. This will be derived from the host for the route that
	// is created for the console.
	ConsoleURL string `json:"consoleURL"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ConsoleList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []Console `json:"items"`
}

// ConsoleAuthentication defines a list of optional configuration for console authentication.
type ConsoleAuthentication struct {
	// An optional, absolute URL to redirect web browsers to after logging out of
	// the console. If not specified, it will redirect to the default login page.
	// This is required when using an identity provider that supports single
	// sign-on (SSO) such as:
	// - OpenID (Keycloak, Azure)
	// - RequestHeader (GSSAPI, SSPI, SAML)
	// - OAuth (GitHub, GitLab, Google)
	// Logging out of the console will destroy the user's token. The logoutRedirect
	// provides the user the option to perform single logout (SLO) through the identity
	// provider to destroy their single sign-on session.
	// +optional
	// +kubebuilder:validation:Pattern=`^$|^((https):\/\/?)[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|\/?))$`
	LogoutRedirect string `json:"logoutRedirect,omitempty"`
}
