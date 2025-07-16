package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +openshift:validation:FeatureGateAwareXValidation:featureGate=ExternalOIDC;ExternalOIDCWithUIDAndExtraClaimMappings,rule="!has(self.spec.oidcProviders) || self.spec.oidcProviders.all(p, !has(p.oidcClients) || p.oidcClients.all(specC, self.status.oidcClients.exists(statusC, statusC.componentNamespace == specC.componentNamespace && statusC.componentName == specC.componentName) || (has(oldSelf.spec.oidcProviders) && oldSelf.spec.oidcProviders.exists(oldP, oldP.name == p.name && has(oldP.oidcClients) && oldP.oidcClients.exists(oldC, oldC.componentNamespace == specC.componentNamespace && oldC.componentName == specC.componentName)))))",message="all oidcClients in the oidcProviders must match their componentName and componentNamespace to either a previously configured oidcClient or they must exist in the status.oidcClients"

// Authentication specifies cluster-wide settings for authentication (like OAuth and
// webhook token authenticators). The canonical name of an instance is `cluster`.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/470
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=authentications,scope=Cluster
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations=release.openshift.io/bootstrap-required=true
type Authentication struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
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

	// oidcProviders are OIDC identity providers that can issue tokens
	// for this cluster
	// Can only be set if "Type" is set to "OIDC".
	//
	// At most one provider can be configured.
	//
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=1
	// +openshift:enable:FeatureGate=ExternalOIDC
	// +openshift:enable:FeatureGate=ExternalOIDCWithUIDAndExtraClaimMappings
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
	// +optional
	IntegratedOAuthMetadata ConfigMapNameReference `json:"integratedOAuthMetadata"`

	// oidcClients is where participating operators place the current OIDC client status
	// for OIDC clients that can be customized by the cluster-admin.
	//
	// +listType=map
	// +listMapKey=componentNamespace
	// +listMapKey=componentName
	// +kubebuilder:validation:MaxItems=20
	// +openshift:enable:FeatureGate=ExternalOIDC
	// +openshift:enable:FeatureGate=ExternalOIDCWithUIDAndExtraClaimMappings
	// +optional
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

// +openshift:validation:FeatureGateAwareEnum:featureGate="",enum="";None;IntegratedOAuth
// +openshift:validation:FeatureGateAwareEnum:featureGate=ExternalOIDC;ExternalOIDCWithUIDAndExtraClaimMappings,enum="";None;IntegratedOAuth;OIDC
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
	// name is a required field that configures the unique human-readable identifier
	// associated with the identity provider.
	// It is used to distinguish between multiple identity providers
	// and has no impact on token validation or authentication mechanics.
	//
	// name must not be an empty string ("").
	//
	// +kubebuilder:validation:MinLength=1
	// +required
	Name string `json:"name"`

	// issuer is a required field that configures how the platform interacts
	// with the identity provider and how tokens issued from the identity provider
	// are evaluated by the Kubernetes API server.
	//
	// +required
	Issuer TokenIssuer `json:"issuer"`

	// oidcClients is an optional field that configures how on-cluster,
	// platform clients should request tokens from the identity provider.
	// oidcClients must not exceed 20 entries and entries must have unique namespace/name pairs.
	//
	// +listType=map
	// +listMapKey=componentNamespace
	// +listMapKey=componentName
	// +kubebuilder:validation:MaxItems=20
	// +optional
	OIDCClients []OIDCClientConfig `json:"oidcClients"`

	// claimMappings is a required field that configures the rules to be used by
	// the Kubernetes API server for translating claims in a JWT token, issued
	// by the identity provider, to a cluster identity.
	//
	// +required
	ClaimMappings TokenClaimMappings `json:"claimMappings"`

	// claimValidationRules is an optional field that configures the rules to
	// be used by the Kubernetes API server for validating the claims in a JWT
	// token issued by the identity provider.
	//
	// Validation rules are joined via an AND operation.
	//
	// +listType=atomic
	// +optional
	ClaimValidationRules []TokenClaimValidationRule `json:"claimValidationRules,omitempty"`
}

// +kubebuilder:validation:MinLength=1
type TokenAudience string

type TokenIssuer struct {
	// issuerURL is a required field that configures the URL used to issue tokens
	// by the identity provider.
	// The Kubernetes API server determines how authentication tokens should be handled
	// by matching the 'iss' claim in the JWT to the issuerURL of configured identity providers.
	//
	// issuerURL must use the 'https' scheme.
	//
	// +kubebuilder:validation:Pattern=`^https:\/\/[^\s]`
	// +required
	URL string `json:"issuerURL"`

	// audiences is a required field that configures the acceptable audiences
	// the JWT token, issued by the identity provider, must be issued to.
	// At least one of the entries must match the 'aud' claim in the JWT token.
	//
	// audiences must contain at least one entry and must not exceed ten entries.
	//
	// +listType=set
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	// +required
	Audiences []TokenAudience `json:"audiences"`

	// issuerCertificateAuthority is an optional field that configures the
	// certificate authority, used by the Kubernetes API server, to validate
	// the connection to the identity provider when fetching discovery information.
	//
	// When not specified, the system trust is used.
	//
	// When specified, it must reference a ConfigMap in the openshift-config
	// namespace containing the PEM-encoded CA certificates under the 'ca-bundle.crt'
	// key in the data field of the ConfigMap.
	//
	// +optional
	CertificateAuthority ConfigMapNameReference `json:"issuerCertificateAuthority"`
}

type TokenClaimMappings struct {
	// username is a required field that configures how the username of a cluster identity
	// should be constructed from the claims in a JWT token issued by the identity provider.
	//
	// +required
	Username UsernameClaimMapping `json:"username"`

	// groups is an optional field that configures how the groups of a cluster identity
	// should be constructed from the claims in a JWT token issued
	// by the identity provider.
	// When referencing a claim, if the claim is present in the JWT
	// token, its value must be a list of groups separated by a comma (',').
	// For example - '"example"' and '"exampleOne", "exampleTwo", "exampleThree"' are valid claim values.
	//
	// +optional
	Groups PrefixedClaimMapping `json:"groups,omitempty"`

	// uid is an optional field for configuring the claim mapping
	// used to construct the uid for the cluster identity.
	//
	// When using uid.claim to specify the claim it must be a single string value.
	// When using uid.expression the expression must result in a single string value.
	//
	// When omitted, this means the user has no opinion and the platform
	// is left to choose a default, which is subject to change over time.
	// The current default is to use the 'sub' claim.
	//
	// +optional
	// +openshift:enable:FeatureGate=ExternalOIDCWithUIDAndExtraClaimMappings
	UID *TokenClaimOrExpressionMapping `json:"uid,omitempty"`

	// extra is an optional field for configuring the mappings
	// used to construct the extra attribute for the cluster identity.
	// When omitted, no extra attributes will be present on the cluster identity.
	// key values for extra mappings must be unique.
	// A maximum of 64 extra attribute mappings may be provided.
	//
	// +optional
	// +kubebuilder:validation:MaxItems=64
	// +listType=map
	// +listMapKey=key
	// +openshift:enable:FeatureGate=ExternalOIDCWithUIDAndExtraClaimMappings
	Extra []ExtraMapping `json:"extra,omitempty"`
}

// TokenClaimMapping allows specifying a JWT token
// claim to be used when mapping claims from an
// authentication token to cluster identities.
type TokenClaimMapping struct {
	// claim is a required field that configures the JWT token
	// claim whose value is assigned to the cluster identity
	// field associated with this mapping.
	//
	// +required
	Claim string `json:"claim"`
}

// TokenClaimOrExpressionMapping allows specifying either a JWT
// token claim or CEL expression to be used when mapping claims
// from an authentication token to cluster identities.
// +kubebuilder:validation:XValidation:rule="has(self.claim) ? !has(self.expression) : has(self.expression)",message="precisely one of claim or expression must be set"
type TokenClaimOrExpressionMapping struct {
	// claim is an optional field for specifying the
	// JWT token claim that is used in the mapping.
	// The value of this claim will be assigned to
	// the field in which this mapping is associated.
	//
	// Precisely one of claim or expression must be set.
	// claim must not be specified when expression is set.
	// When specified, claim must be at least 1 character in length
	// and must not exceed 256 characters in length.
	//
	// +optional
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=1
	Claim string `json:"claim,omitempty"`

	// expression is an optional field for specifying a
	// CEL expression that produces a string value from
	// JWT token claims.
	//
	// CEL expressions have access to the token claims
	// through a CEL variable, 'claims'.
	// 'claims' is a map of claim names to claim values.
	// For example, the 'sub' claim value can be accessed as 'claims.sub'.
	// Nested claims can be accessed using dot notation ('claims.foo.bar').
	//
	// Precisely one of claim or expression must be set.
	// expression must not be specified when claim is set.
	// When specified, expression must be at least 1 character in length
	// and must not exceed 4096 characters in length.
	//
	// +optional
	// +kubebuilder:validation:MaxLength=4096
	// +kubebuilder:validation:MinLength=1
	Expression string `json:"expression,omitempty"`
}

// ExtraMapping allows specifying a key and CEL expression
// to evaluate the keys' value. It is used to create additional
// mappings and attributes added to a cluster identity from
// a provided authentication token.
type ExtraMapping struct {
	// key is a required field that specifies the string
	// to use as the extra attribute key.
	//
	// key must be a domain-prefix path (e.g 'example.org/foo').
	// key must not exceed 510 characters in length.
	// key must contain the '/' character, separating the domain and path characters.
	// key must not be empty.
	//
	// The domain portion of the key (string of characters prior to the '/') must be a valid RFC1123 subdomain.
	// It must not exceed 253 characters in length.
	// It must start and end with an alphanumeric character.
	// It must only contain lower case alphanumeric characters and '-' or '.'.
	// It must not use the reserved domains, or be subdomains of, "kubernetes.io", "k8s.io", and "openshift.io".
	//
	// The path portion of the key (string of characters after the '/') must not be empty and must consist of at least one
	// alphanumeric character, percent-encoded octets, '-', '.', '_', '~', '!', '$', '&', ''', '(', ')', '*', '+', ',', ';', '=', and ':'.
	// It must not exceed 256 characters in length.
	//
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=510
	// +kubebuilder:validation:XValidation:rule="self.contains('/')",message="key must contain the '/' character"
	//
	// +kubebuilder:validation:XValidation:rule="self.split('/', 2)[0].matches(\"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$\")",message="the domain of the key must consist of only lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character"
	// +kubebuilder:validation:XValidation:rule="self.split('/', 2)[0].size() <= 253",message="the domain of the key must not exceed 253 characters in length"
	//
	// +kubebuilder:validation:XValidation:rule="self.split('/', 2)[0] != 'kubernetes.io'",message="the domain 'kubernetes.io' is reserved for Kubernetes use"
	// +kubebuilder:validation:XValidation:rule="!self.split('/', 2)[0].endsWith('.kubernetes.io')",message="the subdomains '*.kubernetes.io' are reserved for Kubernetes use"
	// +kubebuilder:validation:XValidation:rule="self.split('/', 2)[0] != 'k8s.io'",message="the domain 'k8s.io' is reserved for Kubernetes use"
	// +kubebuilder:validation:XValidation:rule="!self.split('/', 2)[0].endsWith('.k8s.io')",message="the subdomains '*.k8s.io' are reserved for Kubernetes use"
	// +kubebuilder:validation:XValidation:rule="self.split('/', 2)[0] != 'openshift.io'",message="the domain 'openshift.io' is reserved for OpenShift use"
	// +kubebuilder:validation:XValidation:rule="!self.split('/', 2)[0].endsWith('.openshift.io')",message="the subdomains '*.openshift.io' are reserved for OpenShift use"
	//
	// +kubebuilder:validation:XValidation:rule="self.split('/', 2)[1].matches('[A-Za-z0-9/\\\\-._~%!$&\\'()*+;=:]+')",message="the path of the key must not be empty and must consist of at least one alphanumeric character, percent-encoded octets, apostrophe, '-', '.', '_', '~', '!', '$', '&', '(', ')', '*', '+', ',', ';', '=', and ':'"
	// +kubebuilder:validation:XValidation:rule="self.split('/', 2)[1].size() <= 256",message="the path of the key must not exceed 256 characters in length"
	Key string `json:"key"`

	// valueExpression is a required field to specify the CEL expression to extract
	// the extra attribute value from a JWT token's claims.
	// valueExpression must produce a string or string array value.
	// "", [], and null are treated as the extra mapping not being present.
	// Empty string values within an array are filtered out.
	//
	// CEL expressions have access to the token claims
	// through a CEL variable, 'claims'.
	// 'claims' is a map of claim names to claim values.
	// For example, the 'sub' claim value can be accessed as 'claims.sub'.
	// Nested claims can be accessed using dot notation ('claims.foo.bar').
	//
	// valueExpression must not exceed 4096 characters in length.
	// valueExpression must not be empty.
	//
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=4096
	ValueExpression string `json:"valueExpression"`
}

// OIDCClientConfig configures how platform clients
// interact with identity providers as an authentication
// method
type OIDCClientConfig struct {
	// componentName is a required field that specifies the name of the platform
	// component being configured to use the identity provider as an authentication mode.
	// It is used in combination with componentNamespace as a unique identifier.
	//
	// componentName must not be an empty string ("") and must not exceed 256 characters in length.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	// +required
	ComponentName string `json:"componentName"`

	// componentNamespace is a required field that specifies the namespace in which the
	// platform component being configured to use the identity provider as an authentication
	// mode is running.
	// It is used in combination with componentName as a unique identifier.
	//
	// componentNamespace must not be an empty string ("") and must not exceed 63 characters in length.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +required
	ComponentNamespace string `json:"componentNamespace"`

	// clientID is a required field that configures the client identifier, from
	// the identity provider, that the platform component uses for authentication
	// requests made to the identity provider.
	// The identity provider must accept this identifier for platform components
	// to be able to use the identity provider as an authentication mode.
	//
	// clientID must not be an empty string ("").
	//
	// +kubebuilder:validation:MinLength=1
	// +required
	ClientID string `json:"clientID"`

	// clientSecret is an optional field that configures the client secret used
	// by the platform component when making authentication requests to the identity provider.
	//
	// When not specified, no client secret will be used when making authentication requests
	// to the identity provider.
	//
	// When specified, clientSecret references a Secret in the 'openshift-config'
	// namespace that contains the client secret in the 'clientSecret' key of the '.data' field.
	// The client secret will be used when making authentication requests to the identity provider.
	//
	// Public clients do not require a client secret but private
	// clients do require a client secret to work with the identity provider.
	//
	// +optional
	ClientSecret SecretNameReference `json:"clientSecret"`

	// extraScopes is an optional field that configures the extra scopes that should
	// be requested by the platform component when making authentication requests to the
	// identity provider.
	// This is useful if you have configured claim mappings that requires specific
	// scopes to be requested beyond the standard OIDC scopes.
	//
	// When omitted, no additional scopes are requested.
	//
	// +listType=set
	// +optional
	ExtraScopes []string `json:"extraScopes"`
}

// OIDCClientStatus represents the current state
// of platform components and how they interact with
// the configured identity providers.
type OIDCClientStatus struct {
	// componentName is a required field that specifies the name of the platform
	// component using the identity provider as an authentication mode.
	// It is used in combination with componentNamespace as a unique identifier.
	//
	// componentName must not be an empty string ("") and must not exceed 256 characters in length.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	// +required
	ComponentName string `json:"componentName"`

	// componentNamespace is a required field that specifies the namespace in which the
	// platform component using the identity provider as an authentication
	// mode is running.
	// It is used in combination with componentName as a unique identifier.
	//
	// componentNamespace must not be an empty string ("") and must not exceed 63 characters in length.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +required
	ComponentNamespace string `json:"componentNamespace"`

	// currentOIDCClients is an optional list of clients that the component is currently using.
	// Entries must have unique issuerURL/clientID pairs.
	//
	// +listType=map
	// +listMapKey=issuerURL
	// +listMapKey=clientID
	// +optional
	CurrentOIDCClients []OIDCClientReference `json:"currentOIDCClients"`

	// consumingUsers is an optional list of ServiceAccounts requiring
	// read permissions on the `clientSecret` secret.
	//
	// consumingUsers must not exceed 5 entries.
	//
	// +kubebuilder:validation:MaxItems=5
	// +listType=set
	// +optional
	ConsumingUsers []ConsumingUser `json:"consumingUsers"`

	// conditions are used to communicate the state of the `oidcClients` entry.
	//
	// Supported conditions include Available, Degraded and Progressing.
	//
	// If Available is true, the component is successfully using the configured client.
	// If Degraded is true, that means something has gone wrong trying to handle the client configuration.
	// If Progressing is true, that means the component is taking some action related to the `oidcClients` entry.
	//
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// OIDCClientReference is a reference to a platform component
// client configuration.
type OIDCClientReference struct {
	// oidcProviderName is a required reference to the 'name' of the identity provider
	// configured in 'oidcProviders' that this client is associated with.
	//
	// oidcProviderName must not be an empty string ("").
	//
	// +kubebuilder:validation:MinLength=1
	// +required
	OIDCProviderName string `json:"oidcProviderName"`

	// issuerURL is a required field that specifies the URL of the identity
	// provider that this client is configured to make requests against.
	//
	// issuerURL must use the 'https' scheme.
	//
	// +kubebuilder:validation:Pattern=`^https:\/\/[^\s]`
	// +required
	IssuerURL string `json:"issuerURL"`

	// clientID is a required field that specifies the client identifier, from
	// the identity provider, that the platform component is using for authentication
	// requests made to the identity provider.
	//
	// clientID must not be empty.
	//
	// +kubebuilder:validation:MinLength=1
	// +required
	ClientID string `json:"clientID"`
}

// +kubebuilder:validation:XValidation:rule="has(self.prefixPolicy) && self.prefixPolicy == 'Prefix' ? (has(self.prefix) && size(self.prefix.prefixString) > 0) : !has(self.prefix)",message="prefix must be set if prefixPolicy is 'Prefix', but must remain unset otherwise"
// +union
type UsernameClaimMapping struct {
	// claim is a required field that configures the JWT token
	// claim whose value is assigned to the cluster identity
	// field associated with this mapping.
	//
	// claim must not be an empty string ("") and must not exceed 256 characters.
	//
	// +required
	// +kubebuilder:validation:MinLength:=1
	// +kubebuilder:validation:MaxLength:=256
	Claim string `json:"claim"`

	// prefixPolicy is an optional field that configures how a prefix should be
	// applied to the value of the JWT claim specified in the 'claim' field.
	//
	// Allowed values are 'Prefix', 'NoPrefix', and omitted (not provided or an empty string).
	//
	// When set to 'Prefix', the value specified in the prefix field will be
	// prepended to the value of the JWT claim.
	// The prefix field must be set when prefixPolicy is 'Prefix'.
	//
	// When set to 'NoPrefix', no prefix will be prepended to the value
	// of the JWT claim.
	//
	// When omitted, this means no opinion and the platform is left to choose
	// any prefixes that are applied which is subject to change over time.
	// Currently, the platform prepends `{issuerURL}#` to the value of the JWT claim
	// when the claim is not 'email'.
	// As an example, consider the following scenario:
	//    `prefix` is unset, `issuerURL` is set to `https://myoidc.tld`,
	//    the JWT claims include "username":"userA" and "email":"userA@myoidc.tld",
	//    and `claim` is set to:
	//    - "username": the mapped value will be "https://myoidc.tld#userA"
	//    - "email": the mapped value will be "userA@myoidc.tld"
	//
	// +kubebuilder:validation:Enum={"", "NoPrefix", "Prefix"}
	// +optional
	// +unionDiscriminator
	PrefixPolicy UsernamePrefixPolicy `json:"prefixPolicy"`

	// prefix configures the prefix that should be prepended to the value
	// of the JWT claim.
	//
	// prefix must be set when prefixPolicy is set to 'Prefix' and must be unset otherwise.
	//
	// +optional
	// +unionMember
	Prefix *UsernamePrefix `json:"prefix"`
}

// UsernamePrefixPolicy configures how prefixes should be applied
// to values extracted from the JWT claims during the process of mapping
// JWT claims to cluster identity attributes.
// +enum
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

// UsernamePrefix configures the string that should
// be used as a prefix for username claim mappings.
type UsernamePrefix struct {
	// prefixString is a required field that configures the prefix that will
	// be applied to cluster identity username attribute
	// during the process of mapping JWT claims to cluster identity attributes.
	//
	// prefixString must not be an empty string ("").
	//
	// +kubebuilder:validation:MinLength=1
	// +required
	PrefixString string `json:"prefixString"`
}

// PrefixedClaimMapping configures a claim mapping
// that allows for an optional prefix.
type PrefixedClaimMapping struct {
	TokenClaimMapping `json:",inline"`

	// prefix is an optional field that configures the prefix that will be
	// applied to the cluster identity attribute during the process of mapping
	// JWT claims to cluster identity attributes.
	//
	// When omitted (""), no prefix is applied to the cluster identity attribute.
	//
	// Example: if `prefix` is set to "myoidc:" and the `claim` in JWT contains
	// an array of strings "a", "b" and  "c", the mapping will result in an
	// array of string "myoidc:a", "myoidc:b" and "myoidc:c".
	//
	// +optional
	Prefix string `json:"prefix"`
}

// TokenValidationRuleType represents the different
// claim validation rule types that can be configured.
// +enum
type TokenValidationRuleType string

const (
	TokenValidationRuleTypeRequiredClaim = "RequiredClaim"
)

type TokenClaimValidationRule struct {
	// type is an optional field that configures the type of the validation rule.
	//
	// Allowed values are 'RequiredClaim' and omitted (not provided or an empty string).
	//
	// When set to 'RequiredClaim', the Kubernetes API server
	// will be configured to validate that the incoming JWT
	// contains the required claim and that its value matches
	// the required value.
	//
	// Defaults to 'RequiredClaim'.
	//
	// +kubebuilder:validation:Enum={"RequiredClaim"}
	// +kubebuilder:default="RequiredClaim"
	Type TokenValidationRuleType `json:"type"`

	// requiredClaim is an optional field that configures the required claim
	// and value that the Kubernetes API server will use to validate if an incoming
	// JWT is valid for this identity provider.
	//
	// +optional
	RequiredClaim *TokenRequiredClaim `json:"requiredClaim,omitempty"`
}

type TokenRequiredClaim struct {
	// claim is a required field that configures the name of the required claim.
	// When taken from the JWT claims, claim must be a string value.
	//
	// claim must not be an empty string ("").
	//
	// +kubebuilder:validation:MinLength=1
	// +required
	Claim string `json:"claim"`

	// requiredValue is a required field that configures the value that 'claim' must
	// have when taken from the incoming JWT claims.
	// If the value in the JWT claims does not match, the token
	// will be rejected for authentication.
	//
	// requiredValue must not be an empty string ("").
	//
	// +kubebuilder:validation:MinLength=1
	// +required
	RequiredValue string `json:"requiredValue"`
}
