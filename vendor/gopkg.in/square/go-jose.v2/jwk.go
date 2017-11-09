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
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"errors"
	"fmt"
	"math/big"
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
	X5c []string `json:"x5c,omitempty"`
}

// JSONWebKey represents a public or private key in JWK format.
type JSONWebKey struct {
	Key          interface{}
	Certificates []*x509.Certificate
	KeyID        string
	Algorithm    string
	Use          string
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

	return json.Marshal(raw)
}

// UnmarshalJSON reads a key from its JSON representation.
func (k *JSONWebKey) UnmarshalJSON(data []byte) (err error) {
	var raw rawJSONWebKey
	err = json.Unmarshal(data, &raw)
	if err != nil {
		return err
	}

	var key interface{}
	switch raw.Kty {
	case "EC":
		if raw.D != nil {
			key, err = raw.ecPrivateKey()
		} else {
			key, err = raw.ecPublicKey()
		}
	case "RSA":
		if raw.D != nil {
			key, err = raw.rsaPrivateKey()
		} else {
			key, err = raw.rsaPublicKey()
		}
	case "oct":
		key, err = raw.symmetricKey()
	case "OKP":
		if raw.Crv == "Ed25519" && raw.X != nil {
			if raw.D != nil {
				key, err = raw.edPrivateKey()
			} else {
				key, err = raw.edPublicKey()
			}
		} else {
			err = fmt.Errorf("square/go-jose: unknown curve %s'", raw.Crv)
		}
	default:
		err = fmt.Errorf("square/go-jose: unknown json web key type '%s'", raw.Kty)
	}

	if err == nil {
		*k = JSONWebKey{Key: key, KeyID: raw.Kid, Algorithm: raw.Alg, Use: raw.Use}
	}

	k.Certificates = make([]*x509.Certificate, len(raw.X5c))
	for i, cert := range raw.X5c {
		raw, err := base64.StdEncoding.DecodeString(cert)
		if err != nil {
			return err
		}
		k.Certificates[i], err = x509.ParseCertificate(raw)
		if err != nil {
			return err
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
		input, err = edThumbprintInput(ed25519.PublicKey(key[0:32]))
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
	case *ecdsa.PublicKey, *rsa.PublicKey, *ed25519.PublicKey:
		return true
	default:
		return false
	}
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
	case *ed25519.PublicKey:
		if len(*key) != 32 {
			return false
		}
	case *ed25519.PrivateKey:
		if len(*key) != 64 {
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
	copy(privateKey[0:32], key.X.bytes())
	copy(privateKey[32:], key.D.bytes())
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
	raw := fromEdPublicKey(ed25519.PublicKey(ed[0:32]))

	raw.D = newBuffer(ed[32:])
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

	raw.D = newBuffer(ec.D.Bytes())

	return raw, nil
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
