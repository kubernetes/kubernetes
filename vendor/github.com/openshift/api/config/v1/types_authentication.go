package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +kubebuilder:subresource:status
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +openshift:validation:FeatureSetAwareXValidation:featureSet=CustomNoUpgrade;TechPreviewNoUpgrade,rule="!has(self.spec.oidcProviders) || self.spec.oidcProviders.all(p, !has(p.oidcClients) || p.oidcClients.all(specC, self.status.oidcClients.exists(statusC, statusC.componentNamespace == specC.componentNamespace && statusC.componentName == specC.componentName) || (has(oldSelf.spec.oidcProviders) && oldSelf.spec.oidcProviders.exists(oldP, oldP.name == p.name && has(oldP.oidcClients) && oldP.oidcClients.exists(oldC, oldC.componentNamespace == specC.componentNamespace && oldC.componentName == specC.componentName)))))",message="all oidcClients in the oidcProviders must match their componentName and componentNamespace to either a previously configured oidcClient or they must exist in the status.oidcClients"

// Authentication specifies cluster-wide settings for authentication (like OAuth and
// webhook token authenticators). The canonical name of an instance is `cluster`.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Authentication struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
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
	// +listType=atomic
	WebhookTokenAuthenticators []DeprecatedWebhookTokenAuthenticator `json:"webhookTokenAuthenticators,omitempty"`

	// webhookTokenAuthenticator configures a remote token reviewer.
	// These remote authentication webhooks can be used to verify bearer tokens
	// via the tokenreviews.authentication.k8s.io REST API. This is required to
	// honor bearer tokens that are provisioned by an external authentication service.
	//
	// Can only be set if "Type" is set to "None".
	//
	// +optional
	WebhookTokenAuthenticator *WebhookTokenAuthenticator `json:"webhookTokenAuthenticator,omitempty"`

	// serviceAccountIssuer is the identifier of the bound service account token
	// issuer.
	// The default is https://kubernetes.default.svc
	// WARNING: Updating this field will not result in immediate invalidation of all bound tokens with the
	// previous issuer value. Instead, the tokens issued by previous service account issuer will continue to
	// be trusted for a time period chosen by the platform (currently set to 24h).
	// This time period is subject to change over time.
	// This allows internal components to transition to use new service account issuer without service distruption.
	// +optional
	ServiceAccountIssuer string `json:"serviceAccountIssuer"`

	// OIDCProviders are OIDC identity providers that can issue tokens
	// for this cluster
	// Can only be set if "Type" is set to "OIDC".
	//
	// At most one provider can be configured.
	//
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=1
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	OIDCProviders []OIDCProvider `json:"oidcProviders,omitempty"`
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

	// OIDCClients is where participating operators place the current OIDC client status
	// for OIDC clients that can be customized by the cluster-admin.
	//
	// +listType=map
	// +listMapKey=componentNamespace
	// +listMapKey=componentName
	// +kubebuilder:validation:MaxItems=20
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	OIDCClients []OIDCClientStatus `json:"oidcClients"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type AuthenticationList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []Authentication `json:"items"`
}

type AuthenticationType string

const (
	// None means that no cluster managed authentication system is in place.
	// Note that user login will only work if a manually configured system is in place and
	// referenced in authentication spec via oauthMetadata and
	// webhookTokenAuthenticator/oidcProviders
	AuthenticationTypeNone AuthenticationType = "None"

	// IntegratedOAuth refers to the cluster managed OAuth server.
	// It is configured via the top level OAuth config.
	AuthenticationTypeIntegratedOAuth AuthenticationType = "IntegratedOAuth"

	// AuthenticationTypeOIDC refers to a configuration with an external
	// OIDC server configured directly with the kube-apiserver.
	AuthenticationTypeOIDC AuthenticationType = "OIDC"
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

type OIDCProvider struct {
	// Name of the OIDC provider
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Required
	// +required
	Name string `json:"name"`
	// Issuer describes atributes of the OIDC token issuer
	//
	// +kubebuilder:validation:Required
	// +required
	Issuer TokenIssuer `json:"issuer"`

	// OIDCClients contains configuration for the platform's clients that
	// need to request tokens from the issuer
	//
	// +listType=map
	// +listMapKey=componentNamespace
	// +listMapKey=componentName
	// +kubebuilder:validation:MaxItems=20
	OIDCClients []OIDCClientConfig `json:"oidcClients"`

	// ClaimMappings describes rules on how to transform information from an
	// ID token into a cluster identity
	ClaimMappings TokenClaimMappings `json:"claimMappings"`

	// ClaimValidationRules are rules that are applied to validate token claims to authenticate users.
	//
	// +listType=atomic
	ClaimValidationRules []TokenClaimValidationRule `json:"claimValidationRules,omitempty"`
}

// +kubebuilder:validation:MinLength=1
type TokenAudience string

type TokenIssuer struct {
	// URL is the serving URL of the token issuer.
	// Must use the https:// scheme.
	//
	// +kubebuilder:validation:Pattern=`^https:\/\/[^\s]`
	// +kubebuilder:validation:Required
	// +required
	URL string `json:"issuerURL"`

	// Audiences is an array of audiences that the token was issued for.
	// Valid tokens must include at least one of these values in their
	// "aud" claim.
	// Must be set to exactly one value.
	//
	// +listType=set
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxItems=1
	// +required
	Audiences []TokenAudience `json:"audiences"`

	// CertificateAuthority is a reference to a config map in the
	// configuration namespace. The .data of the configMap must contain
	// the "ca-bundle.crt" key.
	// If unset, system trust is used instead.
	CertificateAuthority ConfigMapNameReference `json:"issuerCertificateAuthority"`
}

type TokenClaimMappings struct {
	// Username is a name of the claim that should be used to construct
	// usernames for the cluster identity.
	//
	// Default value: "sub"
	Username UsernameClaimMapping `json:"username,omitempty"`

	// Groups is a name of the claim that should be used to construct
	// groups for the cluster identity.
	// The referenced claim must use array of strings values.
	Groups PrefixedClaimMapping `json:"groups,omitempty"`
}

type TokenClaimMapping struct {
	// Claim is a JWT token claim to be used in the mapping
	//
	// +kubebuilder:validation:Required
	// +required
	Claim string `json:"claim"`
}

type OIDCClientConfig struct {
	// ComponentName is the name of the component that is supposed to consume this
	// client configuration
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:Required
	// +required
	ComponentName string `json:"componentName"`

	// ComponentNamespace is the namespace of the component that is supposed to consume this
	// client configuration
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Required
	// +required
	ComponentNamespace string `json:"componentNamespace"`

	// ClientID is the identifier of the OIDC client from the OIDC provider
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Required
	// +required
	ClientID string `json:"clientID"`

	// ClientSecret refers to a secret in the `openshift-config` namespace that
	// contains the client secret in the `clientSecret` key of the `.data` field
	ClientSecret SecretNameReference `json:"clientSecret"`

	// ExtraScopes is an optional set of scopes to request tokens with.
	//
	// +listType=set
	ExtraScopes []string `json:"extraScopes"`
}

type OIDCClientStatus struct {
	// ComponentName is the name of the component that will consume a client configuration.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:Required
	// +required
	ComponentName string `json:"componentName"`

	// ComponentNamespace is the namespace of the component that will consume a client configuration.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Required
	// +required
	ComponentNamespace string `json:"componentNamespace"`

	// CurrentOIDCClients is a list of clients that the component is currently using.
	//
	// +listType=map
	// +listMapKey=issuerURL
	// +listMapKey=clientID
	CurrentOIDCClients []OIDCClientReference `json:"currentOIDCClients"`

	// ConsumingUsers is a slice of ServiceAccounts that need to have read
	// permission on the `clientSecret` secret.
	//
	// +kubebuilder:validation:MaxItems=5
	// +listType=set
	ConsumingUsers []ConsumingUser `json:"consumingUsers"`

	// Conditions are used to communicate the state of the `oidcClients` entry.
	//
	// Supported conditions include Available, Degraded and Progressing.
	//
	// If Available is true, the component is successfully using the configured client.
	// If Degraded is true, that means something has gone wrong trying to handle the client configuration.
	// If Progressing is true, that means the component is taking some action related to the `oidcClients` entry.
	//
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

type OIDCClientReference struct {
	// OIDCName refers to the `name` of the provider from `oidcProviders`
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Required
	// +required
	OIDCProviderName string `json:"oidcProviderName"`

	// URL is the serving URL of the token issuer.
	// Must use the https:// scheme.
	//
	// +kubebuilder:validation:Pattern=`^https:\/\/[^\s]`
	// +kubebuilder:validation:Required
	// +required
	IssuerURL string `json:"issuerURL"`

	// ClientID is the identifier of the OIDC client from the OIDC provider
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Required
	// +required
	ClientID string `json:"clientID"`
}

// +kubebuilder:validation:XValidation:rule="has(self.prefixPolicy) && self.prefixPolicy == 'Prefix' ? (has(self.prefix) && size(self.prefix.prefixString) > 0) : !has(self.prefix)",message="prefix must be set if prefixPolicy is 'Prefix', but must remain unset otherwise"
type UsernameClaimMapping struct {
	TokenClaimMapping `json:",inline"`

	// PrefixPolicy specifies how a prefix should apply.
	//
	// By default, claims other than `email` will be prefixed with the issuer URL to
	// prevent naming clashes with other plugins.
	//
	// Set to "NoPrefix" to disable prefixing.
	//
	// Example:
	//     (1) `prefix` is set to "myoidc:" and `claim` is set to "username".
	//         If the JWT claim `username` contains value `userA`, the resulting
	//         mapped value will be "myoidc:userA".
	//     (2) `prefix` is set to "myoidc:" and `claim` is set to "email". If the
	//         JWT `email` claim contains value "userA@myoidc.tld", the resulting
	//         mapped value will be "myoidc:userA@myoidc.tld".
	//     (3) `prefix` is unset, `issuerURL` is set to `https://myoidc.tld`,
	//         the JWT claims include "username":"userA" and "email":"userA@myoidc.tld",
	//         and `claim` is set to:
	//         (a) "username": the mapped value will be "https://myoidc.tld#userA"
	//         (b) "email": the mapped value will be "userA@myoidc.tld"
	//
	// +kubebuilder:validation:Enum={"", "NoPrefix", "Prefix"}
	PrefixPolicy UsernamePrefixPolicy `json:"prefixPolicy"`

	Prefix *UsernamePrefix `json:"prefix"`
}

type UsernamePrefixPolicy string

var (
	// NoOpinion let's the cluster assign prefixes.  If the username claim is email, there is no prefix
	// If the username claim is anything else, it is prefixed by the issuerURL
	NoOpinion UsernamePrefixPolicy = ""

	// NoPrefix means the username claim value will not have any  prefix
	NoPrefix UsernamePrefixPolicy = "NoPrefix"

	// Prefix means the prefix value must be specified.  It cannot be empty
	Prefix UsernamePrefixPolicy = "Prefix"
)

type UsernamePrefix struct {
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +required
	PrefixString string `json:"prefixString"`
}

type PrefixedClaimMapping struct {
	TokenClaimMapping `json:",inline"`

	// Prefix is a string to prefix the value from the token in the result of the
	// claim mapping.
	//
	// By default, no prefixing occurs.
	//
	// Example: if `prefix` is set to "myoidc:"" and the `claim` in JWT contains
	// an array of strings "a", "b" and  "c", the mapping will result in an
	// array of string "myoidc:a", "myoidc:b" and "myoidc:c".
	Prefix string `json:"prefix"`
}

type TokenValidationRuleType string

const (
	TokenValidationRuleTypeRequiredClaim = "RequiredClaim"
)

type TokenClaimValidationRule struct {
	// Type sets the type of the validation rule
	//
	// +kubebuilder:validation:Enum={"RequiredClaim"}
	// +kubebuilder:default="RequiredClaim"
	Type TokenValidationRuleType `json:"type"`

	// RequiredClaim allows configuring a required claim name and its expected
	// value
	RequiredClaim *TokenRequiredClaim `json:"requiredClaim"`
}

type TokenRequiredClaim struct {
	// Claim is a name of a required claim. Only claims with string values are
	// supported.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Required
	// +required
	Claim string `json:"claim"`

	// RequiredValue is the required value for the claim.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Required
	// +required
	RequiredValue string `json:"requiredValue"`
}
