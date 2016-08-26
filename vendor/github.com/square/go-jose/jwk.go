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
	"fmt"
	"math/big"
	"reflect"
	"strings"
)

// rawJsonWebKey represents a public or private key in JWK format, used for parsing/serializing.
type rawJsonWebKey struct {
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

// JsonWebKey represents a public or private key in JWK format.
type JsonWebKey struct {
	Key          interface{}
	Certificates []*x509.Certificate
	KeyID        string
	Algorithm    string
	Use          string
}

// MarshalJSON serializes the given key to its JSON representation.
func (k JsonWebKey) MarshalJSON() ([]byte, error) {
	var raw *rawJsonWebKey
	var err error

	switch key := k.Key.(type) {
	case *ecdsa.PublicKey:
		raw, err = fromEcPublicKey(key)
	case *rsa.PublicKey:
		raw = fromRsaPublicKey(key)
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

	return MarshalJSON(raw)
}

// UnmarshalJSON reads a key from its JSON representation.
func (k *JsonWebKey) UnmarshalJSON(data []byte) (err error) {
	var raw rawJsonWebKey
	err = UnmarshalJSON(data, &raw)
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
	default:
		err = fmt.Errorf("square/go-jose: unknown json web key type '%s'", raw.Kty)
	}

	if err == nil {
		*k = JsonWebKey{Key: key, KeyID: raw.Kid, Algorithm: raw.Alg, Use: raw.Use}
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

// JsonWebKeySet represents a JWK Set object.
type JsonWebKeySet struct {
	Keys []JsonWebKey `json:"keys"`
}

// Key convenience method returns keys by key ID. Specification states
// that a JWK Set "SHOULD" use distinct key IDs, but allows for some
// cases where they are not distinct. Hence method returns a slice
// of JsonWebKeys.
func (s *JsonWebKeySet) Key(kid string) []JsonWebKey {
	var keys []JsonWebKey
	for _, key := range s.Keys {
		if key.KeyID == kid {
			keys = append(keys, key)
		}
	}

	return keys
}

const rsaThumbprintTemplate = `{"e":"%s","kty":"RSA","n":"%s"}`
const ecThumbprintTemplate = `{"crv":"%s","kty":"EC","x":"%s","y":"%s"}`

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

// Thumbprint computes the JWK Thumbprint of a key using the
// indicated hash algorithm.
func (k *JsonWebKey) Thumbprint(hash crypto.Hash) ([]byte, error) {
	var input string
	var err error
	switch key := k.Key.(type) {
	case *ecdsa.PublicKey:
		input, err = ecThumbprintInput(key.Curve, key.X, key.Y)
	case *ecdsa.PrivateKey:
		input, err = ecThumbprintInput(key.Curve, key.X, key.Y)
	case *rsa.PublicKey:
		input, err = rsaThumbprintInput(key.N, key.E)
	case *rsa.PrivateKey:
		input, err = rsaThumbprintInput(key.N, key.E)
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

// Valid checks that the key contains the expected parameters
func (k *JsonWebKey) Valid() bool {
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
	default:
		return false
	}
	return true
}

func (key rawJsonWebKey) rsaPublicKey() (*rsa.PublicKey, error) {
	if key.N == nil || key.E == nil {
		return nil, fmt.Errorf("square/go-jose: invalid RSA key, missing n/e values")
	}

	return &rsa.PublicKey{
		N: key.N.bigInt(),
		E: key.E.toInt(),
	}, nil
}

func fromRsaPublicKey(pub *rsa.PublicKey) *rawJsonWebKey {
	return &rawJsonWebKey{
		Kty: "RSA",
		N:   newBuffer(pub.N.Bytes()),
		E:   newBufferFromInt(uint64(pub.E)),
	}
}

func (key rawJsonWebKey) ecPublicKey() (*ecdsa.PublicKey, error) {
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
		return nil, fmt.Errorf("square/go-jose: invalid EC key, missing x/y values")
	}

	return &ecdsa.PublicKey{
		Curve: curve,
		X:     key.X.bigInt(),
		Y:     key.Y.bigInt(),
	}, nil
}

func fromEcPublicKey(pub *ecdsa.PublicKey) (*rawJsonWebKey, error) {
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

	key := &rawJsonWebKey{
		Kty: "EC",
		Crv: name,
		X:   newFixedSizeBuffer(xBytes, size),
		Y:   newFixedSizeBuffer(yBytes, size),
	}

	return key, nil
}

func (key rawJsonWebKey) rsaPrivateKey() (*rsa.PrivateKey, error) {
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

func fromRsaPrivateKey(rsa *rsa.PrivateKey) (*rawJsonWebKey, error) {
	if len(rsa.Primes) != 2 {
		return nil, ErrUnsupportedKeyType
	}

	raw := fromRsaPublicKey(&rsa.PublicKey)

	raw.D = newBuffer(rsa.D.Bytes())
	raw.P = newBuffer(rsa.Primes[0].Bytes())
	raw.Q = newBuffer(rsa.Primes[1].Bytes())

	return raw, nil
}

func (key rawJsonWebKey) ecPrivateKey() (*ecdsa.PrivateKey, error) {
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

	return &ecdsa.PrivateKey{
		PublicKey: ecdsa.PublicKey{
			Curve: curve,
			X:     key.X.bigInt(),
			Y:     key.Y.bigInt(),
		},
		D: key.D.bigInt(),
	}, nil
}

func fromEcPrivateKey(ec *ecdsa.PrivateKey) (*rawJsonWebKey, error) {
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

func fromSymmetricKey(key []byte) (*rawJsonWebKey, error) {
	return &rawJsonWebKey{
		Kty: "oct",
		K:   newBuffer(key),
	}, nil
}

func (key rawJsonWebKey) symmetricKey() ([]byte, error) {
	if key.K == nil {
		return nil, fmt.Errorf("square/go-jose: invalid OCT (symmetric) key, missing k value")
	}
	return key.K.bytes(), nil
}
