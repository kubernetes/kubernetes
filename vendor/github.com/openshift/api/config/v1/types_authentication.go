package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Authentication specifies cluster-wide settings for authentication (like OAuth and
// webhook token authenticators). The canonical name of an instance is `cluster`.
type Authentication struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec AuthenticationSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status AuthenticationStatus `json:"status"`
}

type AuthenticationSpec struct {
	// type identifies the cluster managed, user facing authentication mode in use.
	// Specifically, it manages the component that responds to login attempts.
	// The default is IntegratedOAuth.
	// +optional
	Type AuthenticationType `json:"type"`

	// oauthMetadata contains the discovery endpoint data for OAuth 2.0
	// Authorization Server Metadata for an external OAuth server.
	// This discovery document can be viewed from its served location:
	// oc get --raw '/.well-known/oauth-authorization-server'
	// For further details, see the IETF Draft:
	// https://tools.ietf.org/html/draft-ietf-oauth-discovery-04#section-2
	// If oauthMetadata.name is non-empty, this value has precedence
	// over any metadata reference stored in status.
	// The key "oauthMetadata" is used to locate the data.
	// If specified and the config map or expected key is not found, no metadata is served.
	// If the specified metadata is not valid, no metadata is served.
	// The namespace for this config map is openshift-config.
	// +optional
	OAuthMetadata ConfigMapNameReference `json:"oauthMetadata"`

	// webhookTokenAuthenticators is DEPRECATED, setting it has no effect.
	WebhookTokenAuthenticators []DeprecatedWebhookTokenAuthenticator `json:"webhookTokenAuthenticators,omitempty"`

	// webhookTokenAuthenticator configures a remote token reviewer.
	// These remote authentication webhooks can be used to verify bearer tokens
	// via the tokenreviews.authentication.k8s.io REST API. This is required to
	// honor bearer tokens that are provisioned by an external authentication service.
	// +optional
	WebhookTokenAuthenticator *WebhookTokenAuthenticator `json:"webhookTokenAuthenticator,omitempty"`

	// serviceAccountIssuer is the identifier of the bound service account token
	// issuer.
	// The default is https://kubernetes.default.svc
	// WARNING: Updating this field will result in the invalidation of
	// all bound tokens with the previous issuer value. Unless the
	// holder of a bound token has explicit support for a change in
	// issuer, they will not request a new bound token until pod
	// restart or until their existing token exceeds 80% of its
	// duration.
	// +optional
	ServiceAccountIssuer string `json:"serviceAccountIssuer"`
}

type AuthenticationStatus struct {
	// integratedOAuthMetadata contains the discovery endpoint data for OAuth 2.0
	// Authorization Server Metadata for the in-cluster integrated OAuth server.
	// This discovery document can be viewed from its served location:
	// oc get --raw '/.well-known/oauth-authorization-server'
	// For further details, see the IETF Draft:
	// https://tools.ietf.org/html/draft-ietf-oauth-discovery-04#section-2
	// This contains the observed value based on cluster state.
	// An explicitly set value in spec.oauthMetadata has precedence over this field.
	// This field has no meaning if authentication spec.type is not set to IntegratedOAuth.
	// The key "oauthMetadata" is used to locate the data.
	// If the config map or expected key is not found, no metadata is served.
	// If the specified metadata is not valid, no metadata is served.
	// The namespace for this config map is openshift-config-managed.
	IntegratedOAuthMetadata ConfigMapNameReference `json:"integratedOAuthMetadata"`

	// TODO if we add support for an in-cluster operator managed Keycloak instance
	// KeycloakOAuthMetadata ConfigMapNameReference `json:"keycloakOAuthMetadata"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type AuthenticationList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Authentication `json:"items"`
}

type AuthenticationType string

const (
	// None means that no cluster managed authentication system is in place.
	// Note that user login will only work if a manually configured system is in place and
	// referenced in authentication spec via oauthMetadata and webhookTokenAuthenticators.
	AuthenticationTypeNone AuthenticationType = "None"

	// IntegratedOAuth refers to the cluster managed OAuth server.
	// It is configured via the top level OAuth config.
	AuthenticationTypeIntegratedOAuth AuthenticationType = "IntegratedOAuth"

	// TODO if we add support for an in-cluster operator managed Keycloak instance
	// AuthenticationTypeKeycloak AuthenticationType = "Keycloak"
)

// deprecatedWebhookTokenAuthenticator holds the necessary configuration options for a remote token authenticator.
// It's the same as WebhookTokenAuthenticator but it's missing the 'required' validation on KubeConfig field.
type DeprecatedWebhookTokenAuthenticator struct {
	// kubeConfig contains kube config file data which describes how to access the remote webhook service.
	// For further details, see:
	// https://kubernetes.io/docs/reference/access-authn-authz/authentication/#webhook-token-authentication
	// The key "kubeConfig" is used to locate the data.
	// If the secret or expected key is not found, the webhook is not honored.
	// If the specified kube config data is not valid, the webhook is not honored.
	// The namespace for this secret is determined by the point of use.
	KubeConfig SecretNameReference `json:"kubeConfig"`
}

// webhookTokenAuthenticator holds the necessary configuration options for a remote token authenticator
type WebhookTokenAuthenticator struct {
	// kubeConfig references a secret that contains kube config file data which
	// describes how to access the remote webhook service.
	// The namespace for the referenced secret is openshift-config.
	//
	// For further details, see:
	//
	// https://kubernetes.io/docs/reference/access-authn-authz/authentication/#webhook-token-authentication
	//
	// The key "kubeConfig" is used to locate the data.
	// If the secret or expected key is not found, the webhook is not honored.
	// If the specified kube config data is not valid, the webhook is not honored.
	// +kubebuilder:validation:Required
	// +required
	KubeConfig SecretNameReference `json:"kubeConfig"`
}

const (
	// OAuthMetadataKey is the key for the oauth authorization server metadata
	OAuthMetadataKey = "oauthMetadata"

	// KubeConfigKey is the key for the kube config file data in a secret
	KubeConfigKey = "kubeConfig"
)
