package jwt

// Claims represent any form of a JWT Claims Set according to
// https://datatracker.ietf.org/doc/html/rfc7519#section-4. In order to have a
// common basis for validation, it is required that an implementation is able to
// supply at least the claim names provided in
// https://datatracker.ietf.org/doc/html/rfc7519#section-4.1 namely `exp`,
// `iat`, `nbf`, `iss`, `sub` and `aud`.
type Claims interface {
	GetExpirationTime() (*NumericDate, error)
	GetIssuedAt() (*NumericDate, error)
	GetNotBefore() (*NumericDate, error)
	GetIssuer() (string, error)
	GetSubject() (string, error)
	GetAudience() (ClaimStrings, error)
}
