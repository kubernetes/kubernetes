package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=authentications,scope=Cluster
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/475
// +openshift:file-pattern=cvoRunLevel=0000_50,operatorName=authentication,operatorOrdering=01
// +kubebuilder:metadata:annotations=include.release.openshift.io/self-managed-high-availability=true

// Authentication provides information to configure an operator to manage authentication.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Authentication struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// +required
	Spec AuthenticationSpec `json:"spec"`
	// +optional
	Status AuthenticationStatus `json:"status,omitempty"`
}

type AuthenticationSpec struct {
	OperatorSpec `json:",inline"`

	// proxy configures proxy settings for outbound connections made
	// by the authentication stack. When set, it replaces the
	// cluster-wide proxy (proxy.config.openshift.io/cluster)
	// entirely for authentication — individual fields are not
	// inherited from the cluster-wide configuration. When omitted,
	// the cluster-wide proxy is used if configured; otherwise no
	// proxy is used.
	// +openshift:enable:FeatureGate=AuthenticationComponentProxy
	// +optional
	Proxy AuthenticationProxyConfig `json:"proxy,omitzero"`
}

// AuthenticationProxyConfig holds proxy configuration scoped to
// authentication components (the OAuth server and the cluster
// authentication operator).
// +kubebuilder:validation:MinProperties=1
// +kubebuilder:validation:XValidation:rule="has(self.httpProxy) || has(self.httpsProxy)",message="at least one of httpProxy or httpsProxy must be specified"
type AuthenticationProxyConfig struct {
	// httpProxy is the URL of the proxy for HTTP requests.
	// Must be a valid URL with http or https scheme, a non-empty
	// hostname, and no path, query parameters, or fragment.
	// Userinfo (e.g. user:password@host) is allowed for proxy
	// authentication. Maximum length is 2048 characters.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:XValidation:rule="isURL(self)",message="httpProxy must be a valid URL"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || url(self).getScheme() in ['http', 'https']",message="httpProxy must use http or https scheme"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || size(url(self).getHostname()) > 0",message="httpProxy must contain a hostname"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || url(self).getEscapedPath() == '' || url(self).getEscapedPath() == '/'",message="httpProxy must not contain a path"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || url(self).getQuery().size() == 0",message="httpProxy must not contain query parameters"
	// +kubebuilder:validation:XValidation:rule="!self.matches('.*#.*')",message="httpProxy must not contain a fragment"
	// +optional
	HTTPProxy string `json:"httpProxy,omitempty"`

	// httpsProxy is the URL of the proxy for HTTPS requests.
	// Must be a valid URL with http or https scheme, a non-empty
	// hostname, and no path, query parameters, or fragment.
	// Userinfo (e.g. user:password@host) is allowed for proxy
	// authentication. Maximum length is 2048 characters.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:XValidation:rule="isURL(self)",message="httpsProxy must be a valid URL"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || url(self).getScheme() in ['http', 'https']",message="httpsProxy must use http or https scheme"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || size(url(self).getHostname()) > 0",message="httpsProxy must contain a hostname"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || url(self).getEscapedPath() == '' || url(self).getEscapedPath() == '/'",message="httpsProxy must not contain a path"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || url(self).getQuery().size() == 0",message="httpsProxy must not contain query parameters"
	// +kubebuilder:validation:XValidation:rule="!self.matches('.*#.*')",message="httpsProxy must not contain a fragment"
	// +optional
	HTTPSProxy string `json:"httpsProxy,omitempty"`

	// noProxy is a list of hostnames and/or CIDRs and/or IPs for which
	// the proxy should not be used. Must contain at least one entry
	// when set. Each entry must be between 1 and 253 characters long
	// and at most 64 entries are allowed. Duplicate
	// entries are not permitted. Entries that are not valid hostnames,
	// CIDRs, or IPs are silently ignored. Cluster-internal defaults
	// (.cluster.local, .svc, 127.0.0.1, localhost) are always appended
	// automatically and do not need to be included.
	// +listType=set
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=64
	// +kubebuilder:validation:items:MinLength=1
	// +kubebuilder:validation:items:MaxLength=253
	// +optional
	NoProxy []string `json:"noProxy,omitempty"`

	// trustedCA is a reference to a ConfigMap in the openshift-config
	// namespace containing a CA certificate bundle under the key
	// "ca-bundle.crt". This bundle is appended to the system trust store
	// used by authentication components for proxy TLS connections.
	// When omitted, only the system trust store is used.
	// +optional
	TrustedCA AuthenticationConfigMapReference `json:"trustedCA,omitzero"`
}

// AuthenticationConfigMapReference references a ConfigMap in the
// openshift-config namespace.
type AuthenticationConfigMapReference struct {
	// name is the metadata.name of the referenced ConfigMap.
	// Must be a valid DNS subdomain name (RFC 1123): at most 253
	// characters, only lowercase alphanumeric characters, '-' or
	// '.', starting and ending with an alphanumeric character.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="name must be a valid DNS subdomain name: contain no more than 253 characters, contain only lowercase alphanumeric characters, '-' or '.', and start and end with an alphanumeric character"
	// +required
	Name string `json:"name,omitempty"`
}

type AuthenticationStatus struct {
	// oauthAPIServer holds status specific only to oauth-apiserver
	// +optional
	OAuthAPIServer OAuthAPIServerStatus `json:"oauthAPIServer,omitempty"`

	OperatorStatus `json:",inline"`
}

type OAuthAPIServerStatus struct {
	// latestAvailableRevision is the latest revision used as suffix of revisioned
	// secrets like encryption-config. A new revision causes a new deployment of pods.
	// +optional
	// +kubebuilder:validation:Minimum=0
	LatestAvailableRevision int32 `json:"latestAvailableRevision,omitempty"`

	// encryptionStatus contains status reports for the KMS plugin health and its key rotation.
	// +optional
	// +openshift:enable:FeatureGate=KMSEncryption
	EncryptionStatus KMSEncryptionStatus `json:"encryptionStatus,omitempty,omitzero"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// AuthenticationList is a collection of items
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type AuthenticationList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []Authentication `json:"items"`
}
