package oidc

import jose "github.com/go-jose/go-jose/v4"

// JOSE asymmetric signing algorithm values as defined by RFC 7518
//
// see: https://tools.ietf.org/html/rfc7518#section-3.1
const (
	RS256 = "RS256" // RSASSA-PKCS-v1.5 using SHA-256
	RS384 = "RS384" // RSASSA-PKCS-v1.5 using SHA-384
	RS512 = "RS512" // RSASSA-PKCS-v1.5 using SHA-512
	ES256 = "ES256" // ECDSA using P-256 and SHA-256
	ES384 = "ES384" // ECDSA using P-384 and SHA-384
	ES512 = "ES512" // ECDSA using P-521 and SHA-512
	PS256 = "PS256" // RSASSA-PSS using SHA256 and MGF1-SHA256
	PS384 = "PS384" // RSASSA-PSS using SHA384 and MGF1-SHA384
	PS512 = "PS512" // RSASSA-PSS using SHA512 and MGF1-SHA512
	EdDSA = "EdDSA" // Ed25519 using SHA-512
)

var allAlgs = []jose.SignatureAlgorithm{
	jose.RS256,
	jose.RS384,
	jose.RS512,
	jose.ES256,
	jose.ES384,
	jose.ES512,
	jose.PS256,
	jose.PS384,
	jose.PS512,
	jose.EdDSA,
}
