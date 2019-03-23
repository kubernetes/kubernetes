// +build !golint

// Don't lint this file. We don't want to have to add a comment to each constant.

package oidc

const (
	// JOSE asymmetric signing algorithm values as defined by RFC 7518
	//
	// see: https://tools.ietf.org/html/rfc7518#section-3.1
	RS256 = "RS256" // RSASSA-PKCS-v1.5 using SHA-256
	RS384 = "RS384" // RSASSA-PKCS-v1.5 using SHA-384
	RS512 = "RS512" // RSASSA-PKCS-v1.5 using SHA-512
	ES256 = "ES256" // ECDSA using P-256 and SHA-256
	ES384 = "ES384" // ECDSA using P-384 and SHA-384
	ES512 = "ES512" // ECDSA using P-521 and SHA-512
	PS256 = "PS256" // RSASSA-PSS using SHA256 and MGF1-SHA256
	PS384 = "PS384" // RSASSA-PSS using SHA384 and MGF1-SHA384
	PS512 = "PS512" // RSASSA-PSS using SHA512 and MGF1-SHA512
)
