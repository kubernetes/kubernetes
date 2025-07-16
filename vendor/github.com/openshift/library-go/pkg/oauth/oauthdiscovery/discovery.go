package oauthdiscovery

// OauthAuthorizationServerMetadata holds OAuth 2.0 Authorization Server Metadata used for discovery
// https://tools.ietf.org/html/draft-ietf-oauth-discovery-04#section-2
type OauthAuthorizationServerMetadata struct {
	// The authorization server's issuer identifier, which is a URL that uses the https scheme and has no query or fragment components.
	// This is the location where .well-known RFC 5785 [RFC5785] resources containing information about the authorization server are published.
	Issuer string `json:"issuer"`

	// URL of the authorization server's authorization endpoint [RFC6749].
	AuthorizationEndpoint string `json:"authorization_endpoint"`

	// URL of the authorization server's token endpoint [RFC6749].
	TokenEndpoint string `json:"token_endpoint"`

	// JSON array containing a list of the OAuth 2.0 [RFC6749] scope values that this authorization server supports.
	// Servers MAY choose not to advertise some supported scope values even when this parameter is used.
	ScopesSupported []string `json:"scopes_supported"`

	// JSON array containing a list of the OAuth 2.0 response_type values that this authorization server supports.
	// The array values used are the same as those used with the response_types parameter defined by "OAuth 2.0 Dynamic Client Registration Protocol" [RFC7591].
	ResponseTypesSupported []string `json:"response_types_supported"`

	// JSON array containing a list of the OAuth 2.0 grant type values that this authorization server supports.
	// The array values used are the same as those used with the grant_types parameter defined by "OAuth 2.0 Dynamic Client Registration Protocol" [RFC7591].
	GrantTypesSupported []string `json:"grant_types_supported"`

	// JSON array containing a list of PKCE [RFC7636] code challenge methods supported by this authorization server.
	// Code challenge method values are used in the "code_challenge_method" parameter defined in Section 4.3 of [RFC7636].
	// The valid code challenge method values are those registered in the IANA "PKCE Code Challenge Methods" registry [IANA.OAuth.Parameters].
	CodeChallengeMethodsSupported []string `json:"code_challenge_methods_supported"`
}
