package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	configv1 "github.com/openshift/api/config/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type OsinServerConfig struct {
	metav1.TypeMeta `json:",inline"`

	// provides the standard apiserver configuration
	configv1.GenericAPIServerConfig `json:",inline"`

	// oauthConfig holds the necessary configuration options for OAuth authentication
	OAuthConfig OAuthConfig `json:"oauthConfig"`
}

// OAuthConfig holds the necessary configuration options for OAuth authentication
type OAuthConfig struct {
	// masterCA is the CA for verifying the TLS connection back to the MasterURL.
	// This field is deprecated and will be removed in a future release.
	// See loginURL for details.
	// Deprecated
	MasterCA *string `json:"masterCA"`

	// masterURL is used for making server-to-server calls to exchange authorization codes for access tokens
	// This field is deprecated and will be removed in a future release.
	// See loginURL for details.
	// Deprecated
	MasterURL string `json:"masterURL"`

	// masterPublicURL is used for building valid client redirect URLs for internal and external access
	// This field is deprecated and will be removed in a future release.
	// See loginURL for details.
	// Deprecated
	MasterPublicURL string `json:"masterPublicURL"`

	// loginURL, along with masterCA, masterURL and masterPublicURL have distinct
	// meanings depending on how the OAuth server is run.  The two states are:
	// 1. embedded in the kube api server (all 3.x releases)
	// 2. as a standalone external process (all 4.x releases)
	// in the embedded configuration, loginURL is equivalent to masterPublicURL
	// and the other fields have functionality that matches their docs.
	// in the standalone configuration, the fields are used as:
	// loginURL is the URL required to login to the cluster:
	// oc login --server=<loginURL>
	// masterPublicURL is the issuer URL
	// it is accessible from inside (service network) and outside (ingress) of the cluster
	// masterURL is the loopback variation of the token_endpoint URL with no path component
	// it is only accessible from inside (service network) of the cluster
	// masterCA is used to perform TLS verification for connections made to masterURL
	// For further details, see the IETF Draft:
	// https://tools.ietf.org/html/draft-ietf-oauth-discovery-04#section-2
	LoginURL string `json:"loginURL"`

	// assetPublicURL is used for building valid client redirect URLs for external access
	AssetPublicURL string `json:"assetPublicURL"`

	// alwaysShowProviderSelection will force the provider selection page to render even when there is only a single provider.
	AlwaysShowProviderSelection bool `json:"alwaysShowProviderSelection"`

	//identityProviders is an ordered list of ways for a user to identify themselves
	IdentityProviders []IdentityProvider `json:"identityProviders"`

	// grantConfig describes how to handle grants
	GrantConfig GrantConfig `json:"grantConfig"`

	// sessionConfig hold information about configuring sessions.
	SessionConfig *SessionConfig `json:"sessionConfig"`

	// tokenConfig contains options for authorization and access tokens
	TokenConfig TokenConfig `json:"tokenConfig"`

	// templates allow you to customize pages like the login page.
	Templates *OAuthTemplates `json:"templates"`
}

// OAuthTemplates allow for customization of pages like the login page
type OAuthTemplates struct {
	// login is a path to a file containing a go template used to render the login page.
	// If unspecified, the default login page is used.
	Login string `json:"login"`

	// providerSelection is a path to a file containing a go template used to render the provider selection page.
	// If unspecified, the default provider selection page is used.
	ProviderSelection string `json:"providerSelection"`

	// error is a path to a file containing a go template used to render error pages during the authentication or grant flow
	// If unspecified, the default error page is used.
	Error string `json:"error"`
}

// IdentityProvider provides identities for users authenticating using credentials
type IdentityProvider struct {
	// name is used to qualify the identities returned by this provider
	Name string `json:"name"`
	// challenge indicates whether to issue WWW-Authenticate challenges for this provider
	UseAsChallenger bool `json:"challenge"`
	// login indicates whether to use this identity provider for unauthenticated browsers to login against
	UseAsLogin bool `json:"login"`
	// mappingMethod determines how identities from this provider are mapped to users
	MappingMethod string `json:"mappingMethod"`
	// provider contains the information about how to set up a specific identity provider
	// +kubebuilder:pruning:PreserveUnknownFields
	Provider runtime.RawExtension `json:"provider"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BasicAuthPasswordIdentityProvider provides identities for users authenticating using HTTP basic auth credentials
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type BasicAuthPasswordIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`

	// RemoteConnectionInfo contains information about how to connect to the external basic auth server
	configv1.RemoteConnectionInfo `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// AllowAllPasswordIdentityProvider provides identities for users authenticating using non-empty passwords
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type AllowAllPasswordIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DenyAllPasswordIdentityProvider provides no identities for users
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type DenyAllPasswordIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// HTPasswdPasswordIdentityProvider provides identities for users authenticating using htpasswd credentials
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type HTPasswdPasswordIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`

	// file is a reference to your htpasswd file
	File string `json:"file"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LDAPPasswordIdentityProvider provides identities for users authenticating using LDAP credentials
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type LDAPPasswordIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`
	// url is an RFC 2255 URL which specifies the LDAP search parameters to use. The syntax of the URL is
	//    ldap://host:port/basedn?attribute?scope?filter
	URL string `json:"url"`
	// bindDN is an optional DN to bind with during the search phase.
	BindDN string `json:"bindDN"`
	// bindPassword is an optional password to bind with during the search phase.
	BindPassword configv1.StringSource `json:"bindPassword"`

	// insecure, if true, indicates the connection should not use TLS.
	// Cannot be set to true with a URL scheme of "ldaps://"
	// If false, "ldaps://" URLs connect using TLS, and "ldap://" URLs are upgraded to a TLS connection using StartTLS as specified in https://tools.ietf.org/html/rfc2830
	Insecure bool `json:"insecure"`
	// ca is the optional trusted certificate authority bundle to use when making requests to the server
	// If empty, the default system roots are used
	CA string `json:"ca"`
	// attributes maps LDAP attributes to identities
	Attributes LDAPAttributeMapping `json:"attributes"`
}

// LDAPAttributeMapping maps LDAP attributes to OpenShift identity fields
type LDAPAttributeMapping struct {
	// id is the list of attributes whose values should be used as the user ID. Required.
	// LDAP standard identity attribute is "dn"
	ID []string `json:"id"`
	// preferredUsername is the list of attributes whose values should be used as the preferred username.
	// LDAP standard login attribute is "uid"
	PreferredUsername []string `json:"preferredUsername"`
	// name is the list of attributes whose values should be used as the display name. Optional.
	// If unspecified, no display name is set for the identity
	// LDAP standard display name attribute is "cn"
	Name []string `json:"name"`
	// email is the list of attributes whose values should be used as the email address. Optional.
	// If unspecified, no email is set for the identity
	Email []string `json:"email"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KeystonePasswordIdentityProvider provides identities for users authenticating using keystone password credentials
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type KeystonePasswordIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`
	// RemoteConnectionInfo contains information about how to connect to the keystone server
	configv1.RemoteConnectionInfo `json:",inline"`
	// domainName is required for keystone v3
	DomainName string `json:"domainName"`
	// useKeystoneIdentity flag indicates that user should be authenticated by keystone ID, not by username
	UseKeystoneIdentity bool `json:"useKeystoneIdentity"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RequestHeaderIdentityProvider provides identities for users authenticating using request header credentials
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type RequestHeaderIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`

	// loginURL is a URL to redirect unauthenticated /authorize requests to
	// Unauthenticated requests from OAuth clients which expect interactive logins will be redirected here
	// ${url} is replaced with the current URL, escaped to be safe in a query parameter
	//   https://www.example.com/sso-login?then=${url}
	// ${query} is replaced with the current query string
	//   https://www.example.com/auth-proxy/oauth/authorize?${query}
	LoginURL string `json:"loginURL"`

	// challengeURL is a URL to redirect unauthenticated /authorize requests to
	// Unauthenticated requests from OAuth clients which expect WWW-Authenticate challenges will be redirected here
	// ${url} is replaced with the current URL, escaped to be safe in a query parameter
	//   https://www.example.com/sso-login?then=${url}
	// ${query} is replaced with the current query string
	//   https://www.example.com/auth-proxy/oauth/authorize?${query}
	ChallengeURL string `json:"challengeURL"`

	// clientCA is a file with the trusted signer certs.  If empty, no request verification is done, and any direct request to the OAuth server can impersonate any identity from this provider, merely by setting a request header.
	ClientCA string `json:"clientCA"`
	// clientCommonNames is an optional list of common names to require a match from. If empty, any client certificate validated against the clientCA bundle is considered authoritative.
	ClientCommonNames []string `json:"clientCommonNames"`

	// headers is the set of headers to check for identity information
	Headers []string `json:"headers"`
	// preferredUsernameHeaders is the set of headers to check for the preferred username
	PreferredUsernameHeaders []string `json:"preferredUsernameHeaders"`
	// nameHeaders is the set of headers to check for the display name
	NameHeaders []string `json:"nameHeaders"`
	// emailHeaders is the set of headers to check for the email address
	EmailHeaders []string `json:"emailHeaders"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// GitHubIdentityProvider provides identities for users authenticating using GitHub credentials
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type GitHubIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`

	// clientID is the oauth client ID
	ClientID string `json:"clientID"`
	// clientSecret is the oauth client secret
	ClientSecret configv1.StringSource `json:"clientSecret"`
	// organizations optionally restricts which organizations are allowed to log in
	Organizations []string `json:"organizations"`
	// teams optionally restricts which teams are allowed to log in. Format is <org>/<team>.
	Teams []string `json:"teams"`
	// hostname is the optional domain (e.g. "mycompany.com") for use with a hosted instance of GitHub Enterprise.
	// It must match the GitHub Enterprise settings value that is configured at /setup/settings#hostname.
	Hostname string `json:"hostname"`
	// ca is the optional trusted certificate authority bundle to use when making requests to the server.
	// If empty, the default system roots are used.  This can only be configured when hostname is set to a non-empty value.
	CA string `json:"ca"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// GitLabIdentityProvider provides identities for users authenticating using GitLab credentials
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type GitLabIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`

	// ca is the optional trusted certificate authority bundle to use when making requests to the server
	// If empty, the default system roots are used
	CA string `json:"ca"`
	// url is the oauth server base URL
	URL string `json:"url"`
	// clientID is the oauth client ID
	ClientID string `json:"clientID"`
	// clientSecret is the oauth client secret
	ClientSecret configv1.StringSource `json:"clientSecret"`
	// legacy determines if OAuth2 or OIDC should be used
	// If true, OAuth2 is used
	// If false, OIDC is used
	// If nil and the URL's host is gitlab.com, OIDC is used
	// Otherwise, OAuth2 is used
	// In a future release, nil will default to using OIDC
	// Eventually this flag will be removed and only OIDC will be used
	Legacy *bool `json:"legacy,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// GoogleIdentityProvider provides identities for users authenticating using Google credentials
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type GoogleIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`

	// clientID is the oauth client ID
	ClientID string `json:"clientID"`
	// clientSecret is the oauth client secret
	ClientSecret configv1.StringSource `json:"clientSecret"`

	// hostedDomain is the optional Google App domain (e.g. "mycompany.com") to restrict logins to
	HostedDomain string `json:"hostedDomain"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OpenIDIdentityProvider provides identities for users authenticating using OpenID credentials
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type OpenIDIdentityProvider struct {
	metav1.TypeMeta `json:",inline"`

	// ca is the optional trusted certificate authority bundle to use when making requests to the server
	// If empty, the default system roots are used
	CA string `json:"ca"`

	// clientID is the oauth client ID
	ClientID string `json:"clientID"`
	// clientSecret is the oauth client secret
	ClientSecret configv1.StringSource `json:"clientSecret"`

	// extraScopes are any scopes to request in addition to the standard "openid" scope.
	ExtraScopes []string `json:"extraScopes"`

	// extraAuthorizeParameters are any custom parameters to add to the authorize request.
	ExtraAuthorizeParameters map[string]string `json:"extraAuthorizeParameters"`

	// urls to use to authenticate
	URLs OpenIDURLs `json:"urls"`

	// claims mappings
	Claims OpenIDClaims `json:"claims"`
}

// OpenIDURLs are URLs to use when authenticating with an OpenID identity provider
type OpenIDURLs struct {
	// authorize is the oauth authorization URL
	Authorize string `json:"authorize"`
	// token is the oauth token granting URL
	Token string `json:"token"`
	// userInfo is the optional userinfo URL.
	// If present, a granted access_token is used to request claims
	// If empty, a granted id_token is parsed for claims
	UserInfo string `json:"userInfo"`
}

// OpenIDClaims contains a list of OpenID claims to use when authenticating with an OpenID identity provider
type OpenIDClaims struct {
	// id is the list of claims whose values should be used as the user ID. Required.
	// OpenID standard identity claim is "sub"
	ID []string `json:"id"`
	// preferredUsername is the list of claims whose values should be used as the preferred username.
	// If unspecified, the preferred username is determined from the value of the id claim
	PreferredUsername []string `json:"preferredUsername"`
	// name is the list of claims whose values should be used as the display name. Optional.
	// If unspecified, no display name is set for the identity
	Name []string `json:"name"`
	// email is the list of claims whose values should be used as the email address. Optional.
	// If unspecified, no email is set for the identity
	Email []string `json:"email"`
	// groups is the list of claims value of which should be used to synchronize groups
	// from the OIDC provider to OpenShift for the user
	Groups []string `json:"groups"`
}

// GrantConfig holds the necessary configuration options for grant handlers
type GrantConfig struct {
	// method determines the default strategy to use when an OAuth client requests a grant.
	// This method will be used only if the specific OAuth client doesn't provide a strategy
	// of their own. Valid grant handling methods are:
	//  - auto:   always approves grant requests, useful for trusted clients
	//  - prompt: prompts the end user for approval of grant requests, useful for third-party clients
	//  - deny:   always denies grant requests, useful for black-listed clients
	Method GrantHandlerType `json:"method"`

	// serviceAccountMethod is used for determining client authorization for service account oauth client.
	// It must be either: deny, prompt
	ServiceAccountMethod GrantHandlerType `json:"serviceAccountMethod"`
}

type GrantHandlerType string

const (
	// auto auto-approves client authorization grant requests
	GrantHandlerAuto GrantHandlerType = "auto"
	// prompt prompts the user to approve new client authorization grant requests
	GrantHandlerPrompt GrantHandlerType = "prompt"
	// deny auto-denies client authorization grant requests
	GrantHandlerDeny GrantHandlerType = "deny"
)

// SessionConfig specifies options for cookie-based sessions. Used by AuthRequestHandlerSession
type SessionConfig struct {
	// sessionSecretsFile is a reference to a file containing a serialized SessionSecrets object
	// If no file is specified, a random signing and encryption key are generated at each server start
	SessionSecretsFile string `json:"sessionSecretsFile"`
	// sessionMaxAgeSeconds specifies how long created sessions last. Used by AuthRequestHandlerSession
	SessionMaxAgeSeconds int32 `json:"sessionMaxAgeSeconds"`
	// sessionName is the cookie name used to store the session
	SessionName string `json:"sessionName"`
}

// TokenConfig holds the necessary configuration options for authorization and access tokens
type TokenConfig struct {
	// authorizeTokenMaxAgeSeconds defines the maximum age of authorize tokens
	AuthorizeTokenMaxAgeSeconds int32 `json:"authorizeTokenMaxAgeSeconds,omitempty"`
	// accessTokenMaxAgeSeconds defines the maximum age of access tokens
	AccessTokenMaxAgeSeconds int32 `json:"accessTokenMaxAgeSeconds,omitempty"`
	// accessTokenInactivityTimeoutSeconds - DEPRECATED: setting this field has no effect.
	// +optional
	AccessTokenInactivityTimeoutSeconds *int32 `json:"accessTokenInactivityTimeoutSeconds,omitempty"`
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
	// +optional
	AccessTokenInactivityTimeout *metav1.Duration `json:"accessTokenInactivityTimeout,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SessionSecrets list the secrets to use to sign/encrypt and authenticate/decrypt created sessions.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type SessionSecrets struct {
	metav1.TypeMeta `json:",inline"`

	// Secrets is a list of secrets
	// New sessions are signed and encrypted using the first secret.
	// Existing sessions are decrypted/authenticated by each secret until one succeeds. This allows rotating secrets.
	Secrets []SessionSecret `json:"secrets"`
}

// SessionSecret is a secret used to authenticate/decrypt cookie-based sessions
type SessionSecret struct {
	// Authentication is used to authenticate sessions using HMAC. Recommended to use a secret with 32 or 64 bytes.
	Authentication string `json:"authentication"`
	// Encryption is used to encrypt sessions. Must be 16, 24, or 32 characters long, to select AES-128, AES-
	Encryption string `json:"encryption"`
}
