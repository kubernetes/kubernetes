package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// OAuth Server and Identity Provider Config

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OAuth holds cluster-wide information about OAuth.  The canonical name is `cluster`.
// It is used to configure the integrated OAuth server.
// This configuration is only honored when the top level Authentication config has type set to IntegratedOAuth.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/470
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=oauths,scope=Cluster
// +kubebuilder:subresource:status
type OAuth struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata"`
	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec OAuthSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status OAuthStatus `json:"status"`
}

// OAuthSpec contains desired cluster auth configuration
type OAuthSpec struct {
	// identityProviders is an ordered list of ways for a user to identify themselves.
	// When this list is empty, no identities are provisioned for users.
	// +optional
	// +listType=atomic
	IdentityProviders []IdentityProvider `json:"identityProviders,omitempty"`

	// tokenConfig contains options for authorization and access tokens
	TokenConfig TokenConfig `json:"tokenConfig"`

	// templates allow you to customize pages like the login page.
	// +optional
	Templates OAuthTemplates `json:"templates"`
}

// OAuthStatus shows current known state of OAuth server in the cluster
type OAuthStatus struct {
	// TODO Fill in with status of identityProviders and templates (and maybe tokenConfig)
}

// TokenConfig holds the necessary configuration options for authorization and access tokens
type TokenConfig struct {
	// accessTokenMaxAgeSeconds defines the maximum age of access tokens
	AccessTokenMaxAgeSeconds int32 `json:"accessTokenMaxAgeSeconds,omitempty"`

	// accessTokenInactivityTimeoutSeconds - DEPRECATED: setting this field has no effect.
	// +optional
	AccessTokenInactivityTimeoutSeconds int32 `json:"accessTokenInactivityTimeoutSeconds,omitempty"`

	// accessTokenInactivityTimeout defines the token inactivity timeout
	// for tokens granted by any client.
	// The value represents the maximum amount of time that can occur between
	// consecutive uses of the token. Tokens become invalid if they are not
	// used within this temporal window. The user will need to acquire a new
	// token to regain access once a token times out. Takes valid time
	// duration string such as "5m", "1.5h" or "2h45m". The minimum allowed
	// value for duration is 300s (5 minutes). If the timeout is configured
	// per client, then that value takes precedence. If the timeout value is
	// not specified and the client does not override the value, then tokens
	// are valid until their lifetime.
	//
	// WARNING: existing tokens' timeout will not be affected (lowered) by changing this value
	// +optional
	AccessTokenInactivityTimeout *metav1.Duration `json:"accessTokenInactivityTimeout,omitempty"`
}

const (
	// LoginTemplateKey is the key of the login template in a secret
	LoginTemplateKey = "login.html"

	// ProviderSelectionTemplateKey is the key for the provider selection template in a secret
	ProviderSelectionTemplateKey = "providers.html"

	// ErrorsTemplateKey is the key for the errors template in a secret
	ErrorsTemplateKey = "errors.html"

	// BindPasswordKey is the key for the LDAP bind password in a secret
	BindPasswordKey = "bindPassword"

	// ClientSecretKey is the key for the oauth client secret data in a secret
	ClientSecretKey = "clientSecret"

	// HTPasswdDataKey is the key for the htpasswd file data in a secret
	HTPasswdDataKey = "htpasswd"
)

// OAuthTemplates allow for customization of pages like the login page
type OAuthTemplates struct {
	// login is the name of a secret that specifies a go template to use to render the login page.
	// The key "login.html" is used to locate the template data.
	// If specified and the secret or expected key is not found, the default login page is used.
	// If the specified template is not valid, the default login page is used.
	// If unspecified, the default login page is used.
	// The namespace for this secret is openshift-config.
	// +optional
	Login SecretNameReference `json:"login"`

	// providerSelection is the name of a secret that specifies a go template to use to render
	// the provider selection page.
	// The key "providers.html" is used to locate the template data.
	// If specified and the secret or expected key is not found, the default provider selection page is used.
	// If the specified template is not valid, the default provider selection page is used.
	// If unspecified, the default provider selection page is used.
	// The namespace for this secret is openshift-config.
	// +optional
	ProviderSelection SecretNameReference `json:"providerSelection"`

	// error is the name of a secret that specifies a go template to use to render error pages
	// during the authentication or grant flow.
	// The key "errors.html" is used to locate the template data.
	// If specified and the secret or expected key is not found, the default error page is used.
	// If the specified template is not valid, the default error page is used.
	// If unspecified, the default error page is used.
	// The namespace for this secret is openshift-config.
	// +optional
	Error SecretNameReference `json:"error"`
}

// IdentityProvider provides identities for users authenticating using credentials
type IdentityProvider struct {
	// name is used to qualify the identities returned by this provider.
	// - It MUST be unique and not shared by any other identity provider used
	// - It MUST be a valid path segment: name cannot equal "." or ".." or contain "/" or "%" or ":"
	//   Ref: https://godoc.org/github.com/openshift/origin/pkg/user/apis/user/validation#ValidateIdentityProviderName
	Name string `json:"name"`

	// mappingMethod determines how identities from this provider are mapped to users
	// Defaults to "claim"
	// +optional
	MappingMethod MappingMethodType `json:"mappingMethod,omitempty"`

	IdentityProviderConfig `json:",inline"`
}

// MappingMethodType specifies how new identities should be mapped to users when they log in
type MappingMethodType string

const (
	// MappingMethodClaim provisions a user with the identity’s preferred user name. Fails if a user
	// with that user name is already mapped to another identity.
	// Default.
	MappingMethodClaim MappingMethodType = "claim"

	// MappingMethodLookup looks up existing users already mapped to an identity but does not
	// automatically provision users or identities. Requires identities and users be set up
	// manually or using an external process.
	MappingMethodLookup MappingMethodType = "lookup"

	// MappingMethodAdd provisions a user with the identity’s preferred user name. If a user with
	// that user name already exists, the identity is mapped to the existing user, adding to any
	// existing identity mappings for the user.
	MappingMethodAdd MappingMethodType = "add"
)

type IdentityProviderType string

const (
	// IdentityProviderTypeBasicAuth provides identities for users authenticating with HTTP Basic Auth
	IdentityProviderTypeBasicAuth IdentityProviderType = "BasicAuth"

	// IdentityProviderTypeGitHub provides identities for users authenticating using GitHub credentials
	IdentityProviderTypeGitHub IdentityProviderType = "GitHub"

	// IdentityProviderTypeGitLab provides identities for users authenticating using GitLab credentials
	IdentityProviderTypeGitLab IdentityProviderType = "GitLab"

	// IdentityProviderTypeGoogle provides identities for users authenticating using Google credentials
	IdentityProviderTypeGoogle IdentityProviderType = "Google"

	// IdentityProviderTypeHTPasswd provides identities from an HTPasswd file
	IdentityProviderTypeHTPasswd IdentityProviderType = "HTPasswd"

	// IdentityProviderTypeKeystone provides identitities for users authenticating using keystone password credentials
	IdentityProviderTypeKeystone IdentityProviderType = "Keystone"

	// IdentityProviderTypeLDAP provides identities for users authenticating using LDAP credentials
	IdentityProviderTypeLDAP IdentityProviderType = "LDAP"

	// IdentityProviderTypeOpenID provides identities for users authenticating using OpenID credentials
	IdentityProviderTypeOpenID IdentityProviderType = "OpenID"

	// IdentityProviderTypeRequestHeader provides identities for users authenticating using request header credentials
	IdentityProviderTypeRequestHeader IdentityProviderType = "RequestHeader"
)

// IdentityProviderConfig contains configuration for using a specific identity provider
type IdentityProviderConfig struct {
	// type identifies the identity provider type for this entry.
	Type IdentityProviderType `json:"type"`

	// Provider-specific configuration
	// The json tag MUST match the `Type` specified above, case-insensitively
	// e.g. For `Type: "LDAP"`, the `ldap` configuration should be provided

	// basicAuth contains configuration options for the BasicAuth IdP
	// +optional
	BasicAuth *BasicAuthIdentityProvider `json:"basicAuth,omitempty"`

	// github enables user authentication using GitHub credentials
	// +optional
	GitHub *GitHubIdentityProvider `json:"github,omitempty"`

	// gitlab enables user authentication using GitLab credentials
	// +optional
	GitLab *GitLabIdentityProvider `json:"gitlab,omitempty"`

	// google enables user authentication using Google credentials
	// +optional
	Google *GoogleIdentityProvider `json:"google,omitempty"`

	// htpasswd enables user authentication using an HTPasswd file to validate credentials
	// +optional
	HTPasswd *HTPasswdIdentityProvider `json:"htpasswd,omitempty"`

	// keystone enables user authentication using keystone password credentials
	// +optional
	Keystone *KeystoneIdentityProvider `json:"keystone,omitempty"`

	// ldap enables user authentication using LDAP credentials
	// +optional
	LDAP *LDAPIdentityProvider `json:"ldap,omitempty"`

	// openID enables user authentication using OpenID credentials
	// +optional
	OpenID *OpenIDIdentityProvider `json:"openID,omitempty"`

	// requestHeader enables user authentication using request header credentials
	// +optional
	RequestHeader *RequestHeaderIdentityProvider `json:"requestHeader,omitempty"`
}

// BasicAuthPasswordIdentityProvider provides identities for users authenticating using HTTP basic auth credentials
type BasicAuthIdentityProvider struct {
	// OAuthRemoteConnectionInfo contains information about how to connect to the external basic auth server
	OAuthRemoteConnectionInfo `json:",inline"`
}

// OAuthRemoteConnectionInfo holds information necessary for establishing a remote connection
type OAuthRemoteConnectionInfo struct {
	// url is the remote URL to connect to
	URL string `json:"url"`

	// ca is an optional reference to a config map by name containing the PEM-encoded CA bundle.
	// It is used as a trust anchor to validate the TLS certificate presented by the remote server.
	// The key "ca.crt" is used to locate the data.
	// If specified and the config map or expected key is not found, the identity provider is not honored.
	// If the specified ca data is not valid, the identity provider is not honored.
	// If empty, the default system roots are used.
	// The namespace for this config map is openshift-config.
	// +optional
	CA ConfigMapNameReference `json:"ca"`

	// tlsClientCert is an optional reference to a secret by name that contains the
	// PEM-encoded TLS client certificate to present when connecting to the server.
	// The key "tls.crt" is used to locate the data.
	// If specified and the secret or expected key is not found, the identity provider is not honored.
	// If the specified certificate data is not valid, the identity provider is not honored.
	// The namespace for this secret is openshift-config.
	// +optional
	TLSClientCert SecretNameReference `json:"tlsClientCert"`

	// tlsClientKey is an optional reference to a secret by name that contains the
	// PEM-encoded TLS private key for the client certificate referenced in tlsClientCert.
	// The key "tls.key" is used to locate the data.
	// If specified and the secret or expected key is not found, the identity provider is not honored.
	// If the specified certificate data is not valid, the identity provider is not honored.
	// The namespace for this secret is openshift-config.
	// +optional
	TLSClientKey SecretNameReference `json:"tlsClientKey"`
}

// HTPasswdPasswordIdentityProvider provides identities for users authenticating using htpasswd credentials
type HTPasswdIdentityProvider struct {
	// fileData is a required reference to a secret by name containing the data to use as the htpasswd file.
	// The key "htpasswd" is used to locate the data.
	// If the secret or expected key is not found, the identity provider is not honored.
	// If the specified htpasswd data is not valid, the identity provider is not honored.
	// The namespace for this secret is openshift-config.
	FileData SecretNameReference `json:"fileData"`
}

// LDAPPasswordIdentityProvider provides identities for users authenticating using LDAP credentials
type LDAPIdentityProvider struct {
	// url is an RFC 2255 URL which specifies the LDAP search parameters to use.
	// The syntax of the URL is:
	// ldap://host:port/basedn?attribute?scope?filter
	URL string `json:"url"`

	// bindDN is an optional DN to bind with during the search phase.
	// +optional
	BindDN string `json:"bindDN"`

	// bindPassword is an optional reference to a secret by name
	// containing a password to bind with during the search phase.
	// The key "bindPassword" is used to locate the data.
	// If specified and the secret or expected key is not found, the identity provider is not honored.
	// The namespace for this secret is openshift-config.
	// +optional
	BindPassword SecretNameReference `json:"bindPassword"`

	// insecure, if true, indicates the connection should not use TLS
	// WARNING: Should not be set to `true` with the URL scheme "ldaps://" as "ldaps://" URLs always
	//          attempt to connect using TLS, even when `insecure` is set to `true`
	// When `true`, "ldap://" URLS connect insecurely. When `false`, "ldap://" URLs are upgraded to
	// a TLS connection using StartTLS as specified in https://tools.ietf.org/html/rfc2830.
	Insecure bool `json:"insecure"`

	// ca is an optional reference to a config map by name containing the PEM-encoded CA bundle.
	// It is used as a trust anchor to validate the TLS certificate presented by the remote server.
	// The key "ca.crt" is used to locate the data.
	// If specified and the config map or expected key is not found, the identity provider is not honored.
	// If the specified ca data is not valid, the identity provider is not honored.
	// If empty, the default system roots are used.
	// The namespace for this config map is openshift-config.
	// +optional
	CA ConfigMapNameReference `json:"ca"`

	// attributes maps LDAP attributes to identities
	Attributes LDAPAttributeMapping `json:"attributes"`
}

// LDAPAttributeMapping maps LDAP attributes to OpenShift identity fields
type LDAPAttributeMapping struct {
	// id is the list of attributes whose values should be used as the user ID. Required.
	// First non-empty attribute is used. At least one attribute is required. If none of the listed
	// attribute have a value, authentication fails.
	// LDAP standard identity attribute is "dn"
	ID []string `json:"id"`

	// preferredUsername is the list of attributes whose values should be used as the preferred username.
	// LDAP standard login attribute is "uid"
	// +optional
	PreferredUsername []string `json:"preferredUsername,omitempty"`

	// name is the list of attributes whose values should be used as the display name. Optional.
	// If unspecified, no display name is set for the identity
	// LDAP standard display name attribute is "cn"
	// +optional
	Name []string `json:"name,omitempty"`

	// email is the list of attributes whose values should be used as the email address. Optional.
	// If unspecified, no email is set for the identity
	// +optional
	Email []string `json:"email,omitempty"`
}

// KeystonePasswordIdentityProvider provides identities for users authenticating using keystone password credentials
type KeystoneIdentityProvider struct {
	// OAuthRemoteConnectionInfo contains information about how to connect to the keystone server
	OAuthRemoteConnectionInfo `json:",inline"`

	// domainName is required for keystone v3
	DomainName string `json:"domainName"`

	// TODO if we ever add support for 3.11 to 4.0 upgrades, add this configuration
	// useUsernameIdentity indicates that users should be authenticated by username, not keystone ID
	// DEPRECATED - only use this option for legacy systems to ensure backwards compatibility
	// +optional
	// UseUsernameIdentity bool `json:"useUsernameIdentity"`
}

// RequestHeaderIdentityProvider provides identities for users authenticating using request header credentials
type RequestHeaderIdentityProvider struct {
	// loginURL is a URL to redirect unauthenticated /authorize requests to
	// Unauthenticated requests from OAuth clients which expect interactive logins will be redirected here
	// ${url} is replaced with the current URL, escaped to be safe in a query parameter
	//   https://www.example.com/sso-login?then=${url}
	// ${query} is replaced with the current query string
	//   https://www.example.com/auth-proxy/oauth/authorize?${query}
	// Required when login is set to true.
	LoginURL string `json:"loginURL"`

	// challengeURL is a URL to redirect unauthenticated /authorize requests to
	// Unauthenticated requests from OAuth clients which expect WWW-Authenticate challenges will be
	// redirected here.
	// ${url} is replaced with the current URL, escaped to be safe in a query parameter
	//   https://www.example.com/sso-login?then=${url}
	// ${query} is replaced with the current query string
	//   https://www.example.com/auth-proxy/oauth/authorize?${query}
	// Required when challenge is set to true.
	ChallengeURL string `json:"challengeURL"`

	// ca is a required reference to a config map by name containing the PEM-encoded CA bundle.
	// It is used as a trust anchor to validate the TLS certificate presented by the remote server.
	// Specifically, it allows verification of incoming requests to prevent header spoofing.
	// The key "ca.crt" is used to locate the data.
	// If the config map or expected key is not found, the identity provider is not honored.
	// If the specified ca data is not valid, the identity provider is not honored.
	// The namespace for this config map is openshift-config.
	ClientCA ConfigMapNameReference `json:"ca"`

	// clientCommonNames is an optional list of common names to require a match from. If empty, any
	// client certificate validated against the clientCA bundle is considered authoritative.
	// +optional
	ClientCommonNames []string `json:"clientCommonNames,omitempty"`

	// headers is the set of headers to check for identity information
	Headers []string `json:"headers"`

	// preferredUsernameHeaders is the set of headers to check for the preferred username
	PreferredUsernameHeaders []string `json:"preferredUsernameHeaders"`

	// nameHeaders is the set of headers to check for the display name
	NameHeaders []string `json:"nameHeaders"`

	// emailHeaders is the set of headers to check for the email address
	EmailHeaders []string `json:"emailHeaders"`
}

// GitHubIdentityProvider provides identities for users authenticating using GitHub credentials
type GitHubIdentityProvider struct {
	// clientID is the oauth client ID
	ClientID string `json:"clientID"`

	// clientSecret is a required reference to the secret by name containing the oauth client secret.
	// The key "clientSecret" is used to locate the data.
	// If the secret or expected key is not found, the identity provider is not honored.
	// The namespace for this secret is openshift-config.
	ClientSecret SecretNameReference `json:"clientSecret"`

	// organizations optionally restricts which organizations are allowed to log in
	// +optional
	Organizations []string `json:"organizations,omitempty"`

	// teams optionally restricts which teams are allowed to log in. Format is <org>/<team>.
	// +optional
	Teams []string `json:"teams,omitempty"`

	// hostname is the optional domain (e.g. "mycompany.com") for use with a hosted instance of
	// GitHub Enterprise.
	// It must match the GitHub Enterprise settings value configured at /setup/settings#hostname.
	// +optional
	Hostname string `json:"hostname"`

	// ca is an optional reference to a config map by name containing the PEM-encoded CA bundle.
	// It is used as a trust anchor to validate the TLS certificate presented by the remote server.
	// The key "ca.crt" is used to locate the data.
	// If specified and the config map or expected key is not found, the identity provider is not honored.
	// If the specified ca data is not valid, the identity provider is not honored.
	// If empty, the default system roots are used.
	// This can only be configured when hostname is set to a non-empty value.
	// The namespace for this config map is openshift-config.
	// +optional
	CA ConfigMapNameReference `json:"ca"`
}

// GitLabIdentityProvider provides identities for users authenticating using GitLab credentials
type GitLabIdentityProvider struct {
	// clientID is the oauth client ID
	ClientID string `json:"clientID"`

	// clientSecret is a required reference to the secret by name containing the oauth client secret.
	// The key "clientSecret" is used to locate the data.
	// If the secret or expected key is not found, the identity provider is not honored.
	// The namespace for this secret is openshift-config.
	ClientSecret SecretNameReference `json:"clientSecret"`

	// url is the oauth server base URL
	URL string `json:"url"`

	// ca is an optional reference to a config map by name containing the PEM-encoded CA bundle.
	// It is used as a trust anchor to validate the TLS certificate presented by the remote server.
	// The key "ca.crt" is used to locate the data.
	// If specified and the config map or expected key is not found, the identity provider is not honored.
	// If the specified ca data is not valid, the identity provider is not honored.
	// If empty, the default system roots are used.
	// The namespace for this config map is openshift-config.
	// +optional
	CA ConfigMapNameReference `json:"ca"`
}

// GoogleIdentityProvider provides identities for users authenticating using Google credentials
type GoogleIdentityProvider struct {
	// clientID is the oauth client ID
	ClientID string `json:"clientID"`

	// clientSecret is a required reference to the secret by name containing the oauth client secret.
	// The key "clientSecret" is used to locate the data.
	// If the secret or expected key is not found, the identity provider is not honored.
	// The namespace for this secret is openshift-config.
	ClientSecret SecretNameReference `json:"clientSecret"`

	// hostedDomain is the optional Google App domain (e.g. "mycompany.com") to restrict logins to
	// +optional
	HostedDomain string `json:"hostedDomain"`
}

// OpenIDIdentityProvider provides identities for users authenticating using OpenID credentials
type OpenIDIdentityProvider struct {
	// clientID is the oauth client ID
	ClientID string `json:"clientID"`

	// clientSecret is a required reference to the secret by name containing the oauth client secret.
	// The key "clientSecret" is used to locate the data.
	// If the secret or expected key is not found, the identity provider is not honored.
	// The namespace for this secret is openshift-config.
	ClientSecret SecretNameReference `json:"clientSecret"`

	// ca is an optional reference to a config map by name containing the PEM-encoded CA bundle.
	// It is used as a trust anchor to validate the TLS certificate presented by the remote server.
	// The key "ca.crt" is used to locate the data.
	// If specified and the config map or expected key is not found, the identity provider is not honored.
	// If the specified ca data is not valid, the identity provider is not honored.
	// If empty, the default system roots are used.
	// The namespace for this config map is openshift-config.
	// +optional
	CA ConfigMapNameReference `json:"ca"`

	// extraScopes are any scopes to request in addition to the standard "openid" scope.
	// +optional
	ExtraScopes []string `json:"extraScopes,omitempty"`

	// extraAuthorizeParameters are any custom parameters to add to the authorize request.
	// +optional
	ExtraAuthorizeParameters map[string]string `json:"extraAuthorizeParameters,omitempty"`

	// issuer is the URL that the OpenID Provider asserts as its Issuer Identifier.
	// It must use the https scheme with no query or fragment component.
	Issuer string `json:"issuer"`

	// claims mappings
	Claims OpenIDClaims `json:"claims"`
}

// UserIDClaim is the claim used to provide a stable identifier for OIDC identities.
// Per http://openid.net/specs/openid-connect-core-1_0.html#ClaimStability
//
//	"The sub (subject) and iss (issuer) Claims, used together, are the only Claims that an RP can
//	 rely upon as a stable identifier for the End-User, since the sub Claim MUST be locally unique
//	 and never reassigned within the Issuer for a particular End-User, as described in Section 2.
//	 Therefore, the only guaranteed unique identifier for a given End-User is the combination of the
//	 iss Claim and the sub Claim."
const UserIDClaim = "sub"

// OpenIDClaim represents a claim retrieved from an OpenID provider's tokens or userInfo
// responses
// +kubebuilder:validation:MinLength=1
type OpenIDClaim string

// OpenIDClaims contains a list of OpenID claims to use when authenticating with an OpenID identity provider
type OpenIDClaims struct {
	// preferredUsername is the list of claims whose values should be used as the preferred username.
	// If unspecified, the preferred username is determined from the value of the sub claim
	// +listType=atomic
	// +optional
	PreferredUsername []string `json:"preferredUsername,omitempty"`

	// name is the list of claims whose values should be used as the display name. Optional.
	// If unspecified, no display name is set for the identity
	// +listType=atomic
	// +optional
	Name []string `json:"name,omitempty"`

	// email is the list of claims whose values should be used as the email address. Optional.
	// If unspecified, no email is set for the identity
	// +listType=atomic
	// +optional
	Email []string `json:"email,omitempty"`

	// groups is the list of claims value of which should be used to synchronize groups
	// from the OIDC provider to OpenShift for the user.
	// If multiple claims are specified, the first one with a non-empty value is used.
	// +listType=atomic
	// +optional
	Groups []OpenIDClaim `json:"groups,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OAuthList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []OAuth `json:"items"`
}
