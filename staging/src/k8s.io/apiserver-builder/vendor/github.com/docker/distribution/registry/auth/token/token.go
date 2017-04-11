package token

import (
	"crypto"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	log "github.com/Sirupsen/logrus"
	"github.com/docker/libtrust"

	"github.com/docker/distribution/registry/auth"
)

const (
	// TokenSeparator is the value which separates the header, claims, and
	// signature in the compact serialization of a JSON Web Token.
	TokenSeparator = "."
)

// Errors used by token parsing and verification.
var (
	ErrMalformedToken = errors.New("malformed token")
	ErrInvalidToken   = errors.New("invalid token")
)

// ResourceActions stores allowed actions on a named and typed resource.
type ResourceActions struct {
	Type    string   `json:"type"`
	Name    string   `json:"name"`
	Actions []string `json:"actions"`
}

// ClaimSet describes the main section of a JSON Web Token.
type ClaimSet struct {
	// Public claims
	Issuer     string `json:"iss"`
	Subject    string `json:"sub"`
	Audience   string `json:"aud"`
	Expiration int64  `json:"exp"`
	NotBefore  int64  `json:"nbf"`
	IssuedAt   int64  `json:"iat"`
	JWTID      string `json:"jti"`

	// Private claims
	Access []*ResourceActions `json:"access"`
}

// Header describes the header section of a JSON Web Token.
type Header struct {
	Type       string           `json:"typ"`
	SigningAlg string           `json:"alg"`
	KeyID      string           `json:"kid,omitempty"`
	X5c        []string         `json:"x5c,omitempty"`
	RawJWK     *json.RawMessage `json:"jwk,omitempty"`
}

// Token describes a JSON Web Token.
type Token struct {
	Raw       string
	Header    *Header
	Claims    *ClaimSet
	Signature []byte
}

// VerifyOptions is used to specify
// options when verifying a JSON Web Token.
type VerifyOptions struct {
	TrustedIssuers    []string
	AcceptedAudiences []string
	Roots             *x509.CertPool
	TrustedKeys       map[string]libtrust.PublicKey
}

// NewToken parses the given raw token string
// and constructs an unverified JSON Web Token.
func NewToken(rawToken string) (*Token, error) {
	parts := strings.Split(rawToken, TokenSeparator)
	if len(parts) != 3 {
		return nil, ErrMalformedToken
	}

	var (
		rawHeader, rawClaims   = parts[0], parts[1]
		headerJSON, claimsJSON []byte
		err                    error
	)

	defer func() {
		if err != nil {
			log.Errorf("error while unmarshalling raw token: %s", err)
		}
	}()

	if headerJSON, err = joseBase64UrlDecode(rawHeader); err != nil {
		err = fmt.Errorf("unable to decode header: %s", err)
		return nil, ErrMalformedToken
	}

	if claimsJSON, err = joseBase64UrlDecode(rawClaims); err != nil {
		err = fmt.Errorf("unable to decode claims: %s", err)
		return nil, ErrMalformedToken
	}

	token := new(Token)
	token.Header = new(Header)
	token.Claims = new(ClaimSet)

	token.Raw = strings.Join(parts[:2], TokenSeparator)
	if token.Signature, err = joseBase64UrlDecode(parts[2]); err != nil {
		err = fmt.Errorf("unable to decode signature: %s", err)
		return nil, ErrMalformedToken
	}

	if err = json.Unmarshal(headerJSON, token.Header); err != nil {
		return nil, ErrMalformedToken
	}

	if err = json.Unmarshal(claimsJSON, token.Claims); err != nil {
		return nil, ErrMalformedToken
	}

	return token, nil
}

// Verify attempts to verify this token using the given options.
// Returns a nil error if the token is valid.
func (t *Token) Verify(verifyOpts VerifyOptions) error {
	// Verify that the Issuer claim is a trusted authority.
	if !contains(verifyOpts.TrustedIssuers, t.Claims.Issuer) {
		log.Errorf("token from untrusted issuer: %q", t.Claims.Issuer)
		return ErrInvalidToken
	}

	// Verify that the Audience claim is allowed.
	if !contains(verifyOpts.AcceptedAudiences, t.Claims.Audience) {
		log.Errorf("token intended for another audience: %q", t.Claims.Audience)
		return ErrInvalidToken
	}

	// Verify that the token is currently usable and not expired.
	currentUnixTime := time.Now().Unix()
	if !(t.Claims.NotBefore <= currentUnixTime && currentUnixTime <= t.Claims.Expiration) {
		log.Errorf("token not to be used before %d or after %d - currently %d", t.Claims.NotBefore, t.Claims.Expiration, currentUnixTime)
		return ErrInvalidToken
	}

	// Verify the token signature.
	if len(t.Signature) == 0 {
		log.Error("token has no signature")
		return ErrInvalidToken
	}

	// Verify that the signing key is trusted.
	signingKey, err := t.VerifySigningKey(verifyOpts)
	if err != nil {
		log.Error(err)
		return ErrInvalidToken
	}

	// Finally, verify the signature of the token using the key which signed it.
	if err := signingKey.Verify(strings.NewReader(t.Raw), t.Header.SigningAlg, t.Signature); err != nil {
		log.Errorf("unable to verify token signature: %s", err)
		return ErrInvalidToken
	}

	return nil
}

// VerifySigningKey attempts to get the key which was used to sign this token.
// The token header should contain either of these 3 fields:
//      `x5c` - The x509 certificate chain for the signing key. Needs to be
//              verified.
//      `jwk` - The JSON Web Key representation of the signing key.
//              May contain its own `x5c` field which needs to be verified.
//      `kid` - The unique identifier for the key. This library interprets it
//              as a libtrust fingerprint. The key itself can be looked up in
//              the trustedKeys field of the given verify options.
// Each of these methods are tried in that order of preference until the
// signing key is found or an error is returned.
func (t *Token) VerifySigningKey(verifyOpts VerifyOptions) (signingKey libtrust.PublicKey, err error) {
	// First attempt to get an x509 certificate chain from the header.
	var (
		x5c    = t.Header.X5c
		rawJWK = t.Header.RawJWK
		keyID  = t.Header.KeyID
	)

	switch {
	case len(x5c) > 0:
		signingKey, err = parseAndVerifyCertChain(x5c, verifyOpts.Roots)
	case rawJWK != nil:
		signingKey, err = parseAndVerifyRawJWK(rawJWK, verifyOpts)
	case len(keyID) > 0:
		signingKey = verifyOpts.TrustedKeys[keyID]
		if signingKey == nil {
			err = fmt.Errorf("token signed by untrusted key with ID: %q", keyID)
		}
	default:
		err = errors.New("unable to get token signing key")
	}

	return
}

func parseAndVerifyCertChain(x5c []string, roots *x509.CertPool) (leafKey libtrust.PublicKey, err error) {
	if len(x5c) == 0 {
		return nil, errors.New("empty x509 certificate chain")
	}

	// Ensure the first element is encoded correctly.
	leafCertDer, err := base64.StdEncoding.DecodeString(x5c[0])
	if err != nil {
		return nil, fmt.Errorf("unable to decode leaf certificate: %s", err)
	}

	// And that it is a valid x509 certificate.
	leafCert, err := x509.ParseCertificate(leafCertDer)
	if err != nil {
		return nil, fmt.Errorf("unable to parse leaf certificate: %s", err)
	}

	// The rest of the certificate chain are intermediate certificates.
	intermediates := x509.NewCertPool()
	for i := 1; i < len(x5c); i++ {
		intermediateCertDer, err := base64.StdEncoding.DecodeString(x5c[i])
		if err != nil {
			return nil, fmt.Errorf("unable to decode intermediate certificate: %s", err)
		}

		intermediateCert, err := x509.ParseCertificate(intermediateCertDer)
		if err != nil {
			return nil, fmt.Errorf("unable to parse intermediate certificate: %s", err)
		}

		intermediates.AddCert(intermediateCert)
	}

	verifyOpts := x509.VerifyOptions{
		Intermediates: intermediates,
		Roots:         roots,
		KeyUsages:     []x509.ExtKeyUsage{x509.ExtKeyUsageAny},
	}

	// TODO: this call returns certificate chains which we ignore for now, but
	// we should check them for revocations if we have the ability later.
	if _, err = leafCert.Verify(verifyOpts); err != nil {
		return nil, fmt.Errorf("unable to verify certificate chain: %s", err)
	}

	// Get the public key from the leaf certificate.
	leafCryptoKey, ok := leafCert.PublicKey.(crypto.PublicKey)
	if !ok {
		return nil, errors.New("unable to get leaf cert public key value")
	}

	leafKey, err = libtrust.FromCryptoPublicKey(leafCryptoKey)
	if err != nil {
		return nil, fmt.Errorf("unable to make libtrust public key from leaf certificate: %s", err)
	}

	return
}

func parseAndVerifyRawJWK(rawJWK *json.RawMessage, verifyOpts VerifyOptions) (pubKey libtrust.PublicKey, err error) {
	pubKey, err = libtrust.UnmarshalPublicKeyJWK([]byte(*rawJWK))
	if err != nil {
		return nil, fmt.Errorf("unable to decode raw JWK value: %s", err)
	}

	// Check to see if the key includes a certificate chain.
	x5cVal, ok := pubKey.GetExtendedField("x5c").([]interface{})
	if !ok {
		// The JWK should be one of the trusted root keys.
		if _, trusted := verifyOpts.TrustedKeys[pubKey.KeyID()]; !trusted {
			return nil, errors.New("untrusted JWK with no certificate chain")
		}

		// The JWK is one of the trusted keys.
		return
	}

	// Ensure each item in the chain is of the correct type.
	x5c := make([]string, len(x5cVal))
	for i, val := range x5cVal {
		certString, ok := val.(string)
		if !ok || len(certString) == 0 {
			return nil, errors.New("malformed certificate chain")
		}
		x5c[i] = certString
	}

	// Ensure that the x509 certificate chain can
	// be verified up to one of our trusted roots.
	leafKey, err := parseAndVerifyCertChain(x5c, verifyOpts.Roots)
	if err != nil {
		return nil, fmt.Errorf("could not verify JWK certificate chain: %s", err)
	}

	// Verify that the public key in the leaf cert *is* the signing key.
	if pubKey.KeyID() != leafKey.KeyID() {
		return nil, errors.New("leaf certificate public key ID does not match JWK key ID")
	}

	return
}

// accessSet returns a set of actions available for the resource
// actions listed in the `access` section of this token.
func (t *Token) accessSet() accessSet {
	if t.Claims == nil {
		return nil
	}

	accessSet := make(accessSet, len(t.Claims.Access))

	for _, resourceActions := range t.Claims.Access {
		resource := auth.Resource{
			Type: resourceActions.Type,
			Name: resourceActions.Name,
		}

		set, exists := accessSet[resource]
		if !exists {
			set = newActionSet()
			accessSet[resource] = set
		}

		for _, action := range resourceActions.Actions {
			set.add(action)
		}
	}

	return accessSet
}

func (t *Token) compactRaw() string {
	return fmt.Sprintf("%s.%s", t.Raw, joseBase64UrlEncode(t.Signature))
}
