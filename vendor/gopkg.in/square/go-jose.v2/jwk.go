/*-
 * Copyright 2014 Square Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package jose

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"math/big"
	"net/url"
	"reflect"
	"strings"

	"golang.org/x/crypto/ed25519"

	"gopkg.in/square/go-jose.v2/json"
)

// rawJSONWebKey represents a public or private key in JWK format, used for parsing/serializing.
type rawJSONWebKey struct {
	Use string      `json:"use,omitempty"`
	Kty string      `json:"kty,omitempty"`
	Kid string      `json:"kid,omitempty"`
	Crv string      `json:"crv,omitempty"`
	Alg string      `json:"alg,omitempty"`
	K   *byteBuffer `json:"k,omitempty"`
	X   *byteBuffer `json:"x,omitempty"`
	Y   *byteBuffer `json:"y,omitempty"`
	N   *byteBuffer `json:"n,omitempty"`
	E   *byteBuffer `json:"e,omitempty"`
	// -- Following fields are only used for private keys --
	// RSA uses D, P and Q, while ECDSA uses only D. Fields Dp, Dq, and Qi are
	// completely optional. Therefore for RSA/ECDSA, D != nil is a contract that
	// we have a private key whereas D == nil means we have only a public key.
	D  *byteBuffer `json:"d,omitempty"`
	P  *byteBuffer `json:"p,omitempty"`
	Q  *byteBuffer `json:"q,omitempty"`
	Dp *byteBuffer `json:"dp,omitempty"`
	Dq *byteBuffer `json:"dq,omitempty"`
	Qi *byteBuffer `json:"qi,omitempty"`
	// Certificates
	X5c       []string `json:"x5c,omitempty"`
	X5u       *url.URL `json:"x5u,omitempty"`
	X5tSHA1   string   `json:"x5t,omitempty"`
	X5tSHA256 string   `json:"x5t#S256,omitempty"`
}

// JSONWebKey represents a public or private key in JWK format.
type JSONWebKey struct {
	// Cryptographic key, can be a symmetric or asymmetric key.
	Key interface{}
	// Key identifier, parsed from `kid` header.
	KeyID string
	// Key algorithm, parsed from `alg` header.
	Algorithm string
	// Key use, parsed from `use` header.
	Use string

	// X.509 certificate chain, parsed from `x5c` header.
	Certificates []*x509.Certificate
	// X.509 certificate URL, parsed from `x5u` header.
	CertificatesURL *url.URL
	// X.509 certificate thumbprint (SHA-1), parsed from `x5t` header.
	CertificateThumbprintSHA1 []byte
	// X.509 certificate thumbprint (SHA-256), parsed from `x5t#S256` header.
	CertificateThumbprintSHA256 []byte
}

// MarshalJSON serializes the given key to its JSON representation.
func (k JSONWebKey) MarshalJSON() ([]byte, error) {
	var raw *rawJSONWebKey
	var err error

	switch key := k.Key.(type) {
	case ed25519.PublicKey:
		raw = fromEdPublicKey(key)
	case *ecdsa.PublicKey:
		raw, err = fromEcPublicKey(key)
	case *rsa.PublicKey:
		raw = fromRsaPublicKey(key)
	case ed25519.PrivateKey:
		raw, err = fromEdPrivateKey(key)
	case *ecdsa.PrivateKey:
		raw, err = fromEcPrivateKey(key)
	case *rsa.PrivateKey:
		raw, err = fromRsaPrivateKey(key)
	case []byte:
		raw, err = fromSymmetricKey(key)
	default:
		return nil, fmt.Errorf("square/go-jose: unknown key type '%s'", reflect.TypeOf(key))
	}

	if err != nil {
		return nil, err
	}

	raw.Kid = k.KeyID
	raw.Alg = k.Algorithm
	raw.Use = k.Use

	for _, cert := range k.Certificates {
		raw.X5c = append(raw.X5c, base64.StdEncoding.EncodeToString(cert.Raw))
	}

	x5tSHA1Len := len(k.CertificateThumbprintSHA1)
	x5tSHA256Len := len(k.CertificateThumbprintSHA256)
	if x5tSHA1Len > 0 {
		if x5tSHA1Len != sha1.Size {
			return nil, fmt.Errorf("square/go-jose: invalid SHA-1 thumbprint (must be %d bytes, not %d)", sha1.Size, x5tSHA1Len)
		}
		raw.X5tSHA1 = base64.RawURLEncoding.EncodeToString(k.CertificateThumbprintSHA1)
	}
	if x5tSHA256Len > 0 {
		if x5tSHA256Len != sha256.Size {
			return nil, fmt.Errorf("square/go-jose: invalid SHA-256 thumbprint (must be %d bytes, not %d)", sha256.Size, x5tSHA256Len)
		}
		raw.X5tSHA256 = base64.RawURLEncoding.EncodeToString(k.CertificateThumbprintSHA256)
	}

	// If cert chain is attached (as opposed to being behind a URL), check the
	// keys thumbprints to make sure they match what is expected. This is to
	// ensure we don't accidentally produce a JWK with semantically inconsistent
	// data in the headers.
	if len(k.Certificates) > 0 {
		expectedSHA1 := sha1.Sum(k.Certificates[0].Raw)
		expectedSHA256 := sha256.Sum256(k.Certificates[0].Raw)

		if len(k.CertificateThumbprintSHA1) > 0 && !bytes.Equal(k.CertificateThumbprintSHA1, expectedSHA1[:]) {
			return nil, errors.New("square/go-jose: invalid SHA-1 thumbprint, does not match cert chain")
		}
		if len(k.CertificateThumbprintSHA256) > 0 && !bytes.Equal(k.CertificateThumbprintSHA256, expectedSHA256[:]) {
			return nil, errors.New("square/go-jose: invalid or SHA-256 thumbprint, does not match cert chain")
		}
	}

	raw.X5u = k.CertificatesURL

	return json.Marshal(raw)
}

// UnmarshalJSON reads a key from its JSON representation.
func (k *JSONWebKey) UnmarshalJSON(data []byte) (err error) {
	var raw rawJSONWebKey
	err = json.Unmarshal(data, &raw)
	if err != nil {
		return err
	}

	certs, err := parseCertificateChain(raw.X5c)
	if err != nil {
		return fmt.Errorf("square/go-jose: failed to unmarshal x5c field: %s", err)
	}

	var key interface{}
	var certPub interface{}
	var keyPub interface{}

	if len(certs) > 0 {
		// We need to check that leaf public key matches the key embedded in this
		// JWK, as required by the standard (see RFC 7517, Section 4.7). Otherwise
		// the JWK parsed could be semantically invalid. Technically, should also
		// check key usage fields and other extensions on the cert here, but the
		// standard doesn't exactly explain how they're supposed to map from the
		// JWK representation to the X.509 extensions.
		certPub = certs[0].PublicKey
	}

	switch raw.Kty {
	case "EC":
		if raw.D != nil {
			key, err = raw.ecPrivateKey()
			if err == nil {
				keyPub = key.(*ecdsa.PrivateKey).Public()
			}
		} else {
			key, err = raw.ecPublicKey()
			keyPub = key
		}
	case "RSA":
		if raw.D != nil {
			key, err = raw.rsaPrivateKey()
			if err == nil {
				keyPub = key.(*rsa.PrivateKey).Public()
			}
		} else {
			key, err = raw.rsaPublicKey()
			keyPub = key
		}
	case "oct":
		if certPub != nil {
			return errors.New("square/go-jose: invalid JWK, found 'oct' (symmetric) key with cert chain")
		}
		key, err = raw.symmetricKey()
	case "OKP":
		if raw.Crv == "Ed25519" && raw.X != nil {
			if raw.D != nil {
				key, err = raw.edPrivateKey()
				if err == nil {
					keyPub = key.(ed25519.PrivateKey).Public()
				}
			} else {
				key, err = raw.edPublicKey()
				keyPub = key
			}
		} else {
			err = fmt.Errorf("square/go-jose: unknown curve %s'", raw.Crv)
		}
	default:
		err = fmt.Errorf("square/go-jose: unknown json web key type '%s'", raw.Kty)
	}

	if err != nil {
		return
	}

	if certPub != nil && keyPub != nil {
		if !reflect.DeepEqual(certPub, keyPub) {
			return errors.New("square/go-jose: invalid JWK, public keys in key and x5c fields to not match")
		}
	}

	*k = JSONWebKey{Key: key, KeyID: raw.Kid, Algorithm: raw.Alg, Use: raw.Use, Certificates: certs}

	k.CertificatesURL = raw.X5u

	// x5t parameters are base64url-encoded SHA thumbprints
	// See RFC 7517, Section 4.8, https://tools.ietf.org/html/rfc7517#section-4.8
	x5tSHA1bytes, err := base64.RawURLEncoding.DecodeString(raw.X5tSHA1)
	if err != nil {
		return errors.New("square/go-jose: invalid JWK, x5t header has invalid encoding")
	}

	// RFC 7517, Section 4.8 is ambiguous as to whether the digest output should be byte or hex,
	// for this reason, after base64 decoding, if the size is sha1.Size it's likely that the value is a byte encoded
	// checksum so we skip this. Otherwise if the checksum was hex encoded we expect a 40 byte sized array so we'll
	// try to hex decode it. When Marshalling this value we'll always use a base64 encoded version of byte format checksum.
	if len(x5tSHA1bytes) == 2*sha1.Size {
		hx, err := hex.DecodeString(string(x5tSHA1bytes))
		if err != nil {
			return fmt.Errorf("square/go-jose: invalid JWK, unable to hex decode x5t: %v", err)

		}
		x5tSHA1bytes = hx
	}

	k.CertificateThumbprintSHA1 = x5tSHA1bytes

	x5tSHA256bytes, err := base64.RawURLEncoding.DecodeString(raw.X5tSHA256)
	if err != nil {
		return errors.New("square/go-jose: invalid JWK, x5t#S256 header has invalid encoding")
	}

	if len(x5tSHA256bytes) == 2*sha256.Size {
		hx256, err := hex.DecodeString(string(x5tSHA256bytes))
		if err != nil {
			return fmt.Errorf("square/go-jose: invalid JWK, unable to hex decode x5t#S256: %v", err)
		}
		x5tSHA256bytes = hx256
	}

	k.CertificateThumbprintSHA256 = x5tSHA256bytes

	x5tSHA1Len := len(k.CertificateThumbprintSHA1)
	x5tSHA256Len := len(k.CertificateThumbprintSHA256)
	if x5tSHA1Len > 0 && x5tSHA1Len != sha1.Size {
		return errors.New("square/go-jose: invalid JWK, x5t header is of incorrect size")
	}
	if x5tSHA256Len > 0 && x5tSHA256Len != sha256.Size {
		return errors.New("square/go-jose: invalid JWK, x5t#S256 header is of incorrect size")
	}

	// If certificate chain *and* thumbprints are set, verify correctness.
	if len(k.Certificates) > 0 {
		leaf := k.Certificates[0]
		sha1sum := sha1.Sum(leaf.Raw)
		sha256sum := sha256.Sum256(leaf.Raw)

		if len(k.CertificateThumbprintSHA1) > 0 && !bytes.Equal(sha1sum[:], k.CertificateThumbprintSHA1) {
			return errors.New("square/go-jose: invalid JWK, x5c thumbprint does not match x5t value")
		}

		if len(k.CertificateThumbprintSHA256) > 0 && !bytes.Equal(sha256sum[:], k.CertificateThumbprintSHA256) {
			return errors.New("square/go-jose: invalid JWK, x5c thumbprint does not match x5t#S256 value")
		}
	}

	return
}

// JSONWebKeySet represents a JWK Set object.
type JSONWebKeySet struct {
	Keys []JSONWebKey `json:"keys"`
}

// Key convenience method returns keys by key ID. Specification states
// that a JWK Set "SHOULD" use distinct key IDs, but allows for some
// cases where they are not distinct. Hence method returns a slice
// of JSONWebKeys.
func (s *JSONWebKeySet) Key(kid string) []JSONWebKey {
	var keys []JSONWebKey
	for _, key := range s.Keys {
		if key.KeyID == kid {
			keys = append(keys, key)
		}
	}

	return keys
}

const rsaThumbprintTemplate = `{"e":"%s","kty":"RSA","n":"%s"}`
const ecThumbprintTemplate = `{"crv":"%s","kty":"EC","x":"%s","y":"%s"}`
const edThumbprintTemplate = `{"crv":"%s","kty":"OKP",x":"%s"}`

func ecThumbprintInput(curve elliptic.Curve, x, y *big.Int) (string, error) {
	coordLength := curveSize(curve)
	crv, err := curveName(curve)
	if err != nil {
		return "", err
	}

	if len(x.Bytes()) > coordLength || len(y.Bytes()) > coordLength {
		return "", errors.New("square/go-jose: invalid elliptic key (too large)")
	}

	return fmt.Sprintf(ecThumbprintTemplate, crv,
		newFixedSizeBuffer(x.Bytes(), coordLength).base64(),
		newFixedSizeBuffer(y.Bytes(), coordLength).base64()), nil
}

func rsaThumbprintInput(n *big.Int, e int) (string, error) {
	return fmt.Sprintf(rsaThumbprintTemplate,
		newBufferFromInt(uint64(e)).base64(),
		newBuffer(n.Bytes()).base64()), nil
}

func edThumbprintInput(ed ed25519.PublicKey) (string, error) {
	crv := "Ed25519"
	if len(ed) > 32 {
		return "", errors.New("square/go-jose: invalid elliptic key (too large)")
	}
	return fmt.Sprintf(edThumbprintTemplate, crv,
		newFixedSizeBuffer(ed, 32).base64()), nil
}

// Thumbprint computes the JWK Thumbprint of a key using the
// indicated hash algorithm.
func (k *JSONWebKey) Thumbprint(hash crypto.Hash) ([]byte, error) {
	var input string
	var err error
	switch key := k.Key.(type) {
	case ed25519.PublicKey:
		input, err = edThumbprintInput(key)
	case *ecdsa.PublicKey:
		input, err = ecThumbprintInput(key.Curve, key.X, key.Y)
	case *ecdsa.PrivateKey:
		input, err = ecThumbprintInput(key.Curve, key.X, key.Y)
	case *rsa.PublicKey:
		input, err = rsaThumbprintInput(key.N, key.E)
	case *rsa.PrivateKey:
		input, err = rsaThumbprintInput(key.N, key.E)
	case ed25519.PrivateKey:
		input, err = edThumbprintInput(ed25519.PublicKey(key[32:]))
	default:
		return nil, fmt.Errorf("square/go-jose: unknown key type '%s'", reflect.TypeOf(key))
	}

	if err != nil {
		return nil, err
	}

	h := hash.New()
	h.Write([]byte(input))
	return h.Sum(nil), nil
}

// IsPublic returns true if the JWK represents a public key (not symmetric, not private).
func (k *JSONWebKey) IsPublic() bool {
	switch k.Key.(type) {
	case *ecdsa.PublicKey, *rsa.PublicKey, ed25519.PublicKey:
		return true
	default:
		return false
	}
}

// Public creates JSONWebKey with corresponding publik key if JWK represents asymmetric private key.
func (k *JSONWebKey) Public() JSONWebKey {
	if k.IsPublic() {
		return *k
	}
	ret := *k
	switch key := k.Key.(type) {
	case *ecdsa.PrivateKey:
		ret.Key = key.Public()
	case *rsa.PrivateKey:
		ret.Key = key.Public()
	case ed25519.PrivateKey:
		ret.Key = key.Public()
	default:
		return JSONWebKey{} // returning invalid key
	}
	return ret
}

// Valid checks that the key contains the expected parameters.
func (k *JSONWebKey) Valid() bool {
	if k.Key == nil {
		return false
	}
	switch key := k.Key.(type) {
	case *ecdsa.PublicKey:
		if key.Curve == nil || key.X == nil || key.Y == nil {
			return false
		}
	case *ecdsa.PrivateKey:
		if key.Curve == nil || key.X == nil || key.Y == nil || key.D == nil {
			return false
		}
	case *rsa.PublicKey:
		if key.N == nil || key.E == 0 {
			return false
		}
	case *rsa.PrivateKey:
		if key.N == nil || key.E == 0 || key.D == nil || len(key.Primes) < 2 {
			return false
		}
	case ed25519.PublicKey:
		if len(key) != 32 {
			return false
		}
	case ed25519.PrivateKey:
		if len(key) != 64 {
			return false
		}
	default:
		return false
	}
	return true
}

func (key rawJSONWebKey) rsaPublicKey() (*rsa.PublicKey, error) {
	if key.N == nil || key.E == nil {
		return nil, fmt.Errorf("square/go-jose: invalid RSA key, missing n/e values")
	}

	return &rsa.PublicKey{
		N: key.N.bigInt(),
		E: key.E.toInt(),
	}, nil
}

func fromEdPublicKey(pub ed25519.PublicKey) *rawJSONWebKey {
	return &rawJSONWebKey{
		Kty: "OKP",
		Crv: "Ed25519",
		X:   newBuffer(pub),
	}
}

func fromRsaPublicKey(pub *rsa.PublicKey) *rawJSONWebKey {
	return &rawJSONWebKey{
		Kty: "RSA",
		N:   newBuffer(pub.N.Bytes()),
		E:   newBufferFromInt(uint64(pub.E)),
	}
}

func (key rawJSONWebKey) ecPublicKey() (*ecdsa.PublicKey, error) {
	var curve elliptic.Curve
	switch key.Crv {
	case "P-256":
		curve = elliptic.P256()
	case "P-384":
		curve = elliptic.P384()
	case "P-521":
		curve = elliptic.P521()
	default:
		return nil, fmt.Errorf("square/go-jose: unsupported elliptic curve '%s'", key.Crv)
	}

	if key.X == nil || key.Y == nil {
		return nil, errors.New("square/go-jose: invalid EC key, missing x/y values")
	}

	// The length of this octet string MUST be the full size of a coordinate for
	// the curve specified in the "crv" parameter.
	// https://tools.ietf.org/html/rfc7518#section-6.2.1.2
	if curveSize(curve) != len(key.X.data) {
		return nil, fmt.Errorf("square/go-jose: invalid EC public key, wrong length for x")
	}

	if curveSize(curve) != len(key.Y.data) {
		return nil, fmt.Errorf("square/go-jose: invalid EC public key, wrong length for y")
	}

	x := key.X.bigInt()
	y := key.Y.bigInt()

	if !curve.IsOnCurve(x, y) {
		return nil, errors.New("square/go-jose: invalid EC key, X/Y are not on declared curve")
	}

	return &ecdsa.PublicKey{
		Curve: curve,
		X:     x,
		Y:     y,
	}, nil
}

func fromEcPublicKey(pub *ecdsa.PublicKey) (*rawJSONWebKey, error) {
	if pub == nil || pub.X == nil || pub.Y == nil {
		return nil, fmt.Errorf("square/go-jose: invalid EC key (nil, or X/Y missing)")
	}

	name, err := curveName(pub.Curve)
	if err != nil {
		return nil, err
	}

	size := curveSize(pub.Curve)

	xBytes := pub.X.Bytes()
	yBytes := pub.Y.Bytes()

	if len(xBytes) > size || len(yBytes) > size {
		return nil, fmt.Errorf("square/go-jose: invalid EC key (X/Y too large)")
	}

	key := &rawJSONWebKey{
		Kty: "EC",
		Crv: name,
		X:   newFixedSizeBuffer(xBytes, size),
		Y:   newFixedSizeBuffer(yBytes, size),
	}

	return key, nil
}

func (key rawJSONWebKey) edPrivateKey() (ed25519.PrivateKey, error) {
	var missing []string
	switch {
	case key.D == nil:
		missing = append(missing, "D")
	case key.X == nil:
		missing = append(missing, "X")
	}

	if len(missing) > 0 {
		return nil, fmt.Errorf("square/go-jose: invalid Ed25519 private key, missing %s value(s)", strings.Join(missing, ", "))
	}

	privateKey := make([]byte, ed25519.PrivateKeySize)
	copy(privateKey[0:32], key.D.bytes())
	copy(privateKey[32:], key.X.bytes())
	rv := ed25519.PrivateKey(privateKey)
	return rv, nil
}

func (key rawJSONWebKey) edPublicKey() (ed25519.PublicKey, error) {
	if key.X == nil {
		return nil, fmt.Errorf("square/go-jose: invalid Ed key, missing x value")
	}
	publicKey := make([]byte, ed25519.PublicKeySize)
	copy(publicKey[0:32], key.X.bytes())
	rv := ed25519.PublicKey(publicKey)
	return rv, nil
}

func (key rawJSONWebKey) rsaPrivateKey() (*rsa.PrivateKey, error) {
	var missing []string
	switch {
	case key.N == nil:
		missing = append(missing, "N")
	case key.E == nil:
		missing = append(missing, "E")
	case key.D == nil:
		missing = append(missing, "D")
	case key.P == nil:
		missing = append(missing, "P")
	case key.Q == nil:
		missing = append(missing, "Q")
	}

	if len(missing) > 0 {
		return nil, fmt.Errorf("square/go-jose: invalid RSA private key, missing %s value(s)", strings.Join(missing, ", "))
	}

	rv := &rsa.PrivateKey{
		PublicKey: rsa.PublicKey{
			N: key.N.bigInt(),
			E: key.E.toInt(),
		},
		D: key.D.bigInt(),
		Primes: []*big.Int{
			key.P.bigInt(),
			key.Q.bigInt(),
		},
	}

	if key.Dp != nil {
		rv.Precomputed.Dp = key.Dp.bigInt()
	}
	if key.Dq != nil {
		rv.Precomputed.Dq = key.Dq.bigInt()
	}
	if key.Qi != nil {
		rv.Precomputed.Qinv = key.Qi.bigInt()
	}

	err := rv.Validate()
	return rv, err
}

func fromEdPrivateKey(ed ed25519.PrivateKey) (*rawJSONWebKey, error) {
	raw := fromEdPublicKey(ed25519.PublicKey(ed[32:]))

	raw.D = newBuffer(ed[0:32])
	return raw, nil
}

func fromRsaPrivateKey(rsa *rsa.PrivateKey) (*rawJSONWebKey, error) {
	if len(rsa.Primes) != 2 {
		return nil, ErrUnsupportedKeyType
	}

	raw := fromRsaPublicKey(&rsa.PublicKey)

	raw.D = newBuffer(rsa.D.Bytes())
	raw.P = newBuffer(rsa.Primes[0].Bytes())
	raw.Q = newBuffer(rsa.Primes[1].Bytes())

	if rsa.Precomputed.Dp != nil {
		raw.Dp = newBuffer(rsa.Precomputed.Dp.Bytes())
	}
	if rsa.Precomputed.Dq != nil {
		raw.Dq = newBuffer(rsa.Precomputed.Dq.Bytes())
	}
	if rsa.Precomputed.Qinv != nil {
		raw.Qi = newBuffer(rsa.Precomputed.Qinv.Bytes())
	}

	return raw, nil
}

func (key rawJSONWebKey) ecPrivateKey() (*ecdsa.PrivateKey, error) {
	var curve elliptic.Curve
	switch key.Crv {
	case "P-256":
		curve = elliptic.P256()
	case "P-384":
		curve = elliptic.P384()
	case "P-521":
		curve = elliptic.P521()
	default:
		return nil, fmt.Errorf("square/go-jose: unsupported elliptic curve '%s'", key.Crv)
	}

	if key.X == nil || key.Y == nil || key.D == nil {
		return nil, fmt.Errorf("square/go-jose: invalid EC private key, missing x/y/d values")
	}

	// The length of this octet string MUST be the full size of a coordinate for
	// the curve specified in the "crv" parameter.
	// https://tools.ietf.org/html/rfc7518#section-6.2.1.2
	if curveSize(curve) != len(key.X.data) {
		return nil, fmt.Errorf("square/go-jose: invalid EC private key, wrong length for x")
	}

	if curveSize(curve) != len(key.Y.data) {
		return nil, fmt.Errorf("square/go-jose: invalid EC private key, wrong length for y")
	}

	// https://tools.ietf.org/html/rfc7518#section-6.2.2.1
	if dSize(curve) != len(key.D.data) {
		return nil, fmt.Errorf("square/go-jose: invalid EC private key, wrong length for d")
	}

	x := key.X.bigInt()
	y := key.Y.bigInt()

	if !curve.IsOnCurve(x, y) {
		return nil, errors.New("square/go-jose: invalid EC key, X/Y are not on declared curve")
	}

	return &ecdsa.PrivateKey{
		PublicKey: ecdsa.PublicKey{
			Curve: curve,
			X:     x,
			Y:     y,
		},
		D: key.D.bigInt(),
	}, nil
}

func fromEcPrivateKey(ec *ecdsa.PrivateKey) (*rawJSONWebKey, error) {
	raw, err := fromEcPublicKey(&ec.PublicKey)
	if err != nil {
		return nil, err
	}

	if ec.D == nil {
		return nil, fmt.Errorf("square/go-jose: invalid EC private key")
	}

	raw.D = newFixedSizeBuffer(ec.D.Bytes(), dSize(ec.PublicKey.Curve))

	return raw, nil
}

// dSize returns the size in octets for the "d" member of an elliptic curve
// private key.
// The length of this octet string MUST be ceiling(log-base-2(n)/8)
// octets (where n is the order of the curve).
// https://tools.ietf.org/html/rfc7518#section-6.2.2.1
func dSize(curve elliptic.Curve) int {
	order := curve.Params().P
	bitLen := order.BitLen()
	size := bitLen / 8
	if bitLen%8 != 0 {
		size = size + 1
	}
	return size
}

func fromSymmetricKey(key []byte) (*rawJSONWebKey, error) {
	return &rawJSONWebKey{
		Kty: "oct",
		K:   newBuffer(key),
	}, nil
}

func (key rawJSONWebKey) symmetricKey() ([]byte, error) {
	if key.K == nil {
		return nil, fmt.Errorf("square/go-jose: invalid OCT (symmetric) key, missing k value")
	}
	return key.K.bytes(), nil
}
