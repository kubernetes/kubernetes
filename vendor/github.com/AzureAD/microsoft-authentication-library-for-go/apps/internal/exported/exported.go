// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// package exported contains internal types that are re-exported from a public package
package exported

// AssertionRequestOptions has information required to generate a client assertion
type AssertionRequestOptions struct {
	// ClientID identifies the application for which an assertion is requested. Used as the assertion's "iss" and "sub" claims.
	ClientID string

	// TokenEndpoint is the intended token endpoint. Used as the assertion's "aud" claim.
	TokenEndpoint string
}

// TokenProviderParameters is the authentication parameters passed to token providers
type TokenProviderParameters struct {
	// Claims contains any additional claims requested for the token
	Claims string
	// CorrelationID of the authentication request
	CorrelationID string
	// Scopes requested for the token
	Scopes []string
	// TenantID identifies the tenant in which to authenticate
	TenantID string
}

// TokenProviderResult is the authentication result returned by custom token providers
type TokenProviderResult struct {
	// AccessToken is the requested token
	AccessToken string
	// ExpiresInSeconds is the lifetime of the token in seconds
	ExpiresInSeconds int
}
