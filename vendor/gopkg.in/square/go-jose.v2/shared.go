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
	"crypto/elliptic"
	"crypto/x509"
	"encoding/base64"
	"errors"
	"fmt"

	"gopkg.in/square/go-jose.v2/json"
)

// KeyAlgorithm represents a key management algorithm.
type KeyAlgorithm string

// SignatureAlgorithm represents a signature (or MAC) algorithm.
type SignatureAlgorithm string

// ContentEncryption represents a content encryption algorithm.
type ContentEncryption string

// CompressionAlgorithm represents an algorithm used for plaintext compression.
type CompressionAlgorithm string

// ContentType represents type of the contained data.
type ContentType string

var (
	// ErrCryptoFailure represents an error in cryptographic primitive. This
	// occurs when, for example, a message had an invalid authentication tag or
	// could not be decrypted.
	ErrCryptoFailure = errors.New("square/go-jose: error in cryptographic primitive")

	// ErrUnsupportedAlgorithm indicates that a selected algorithm is not
	// supported. This occurs when trying to instantiate an encrypter for an
	// algorithm that is not yet implemented.
	ErrUnsupportedAlgorithm = errors.New("square/go-jose: unknown/unsupported algorithm")

	// ErrUnsupportedKeyType indicates that the given key type/format is not
	// supported. This occurs when trying to instantiate an encrypter and passing
	// it a key of an unrecognized type or with unsupported parameters, such as
	// an RSA private key with more than two primes.
	ErrUnsupportedKeyType = errors.New("square/go-jose: unsupported key type/format")

	// ErrInvalidKeySize indicates that the given key is not the correct size
	// for the selected algorithm. This can occur, for example, when trying to
	// encrypt with AES-256 but passing only a 128-bit key as input.
	ErrInvalidKeySize = errors.New("square/go-jose: invalid key size for algorithm")

	// ErrNotSupported serialization of object is not supported. This occurs when
	// trying to compact-serialize an object which can't be represented in
	// compact form.
	ErrNotSupported = errors.New("square/go-jose: compact serialization not supported for object")

	// ErrUnprotectedNonce indicates that while parsing a JWS or JWE object, a
	// nonce header parameter was included in an unprotected header object.
	ErrUnprotectedNonce = errors.New("square/go-jose: Nonce parameter included in unprotected header")
)

// Key management algorithms
const (
	ED25519            = KeyAlgorithm("ED25519")
	RSA1_5             = KeyAlgorithm("RSA1_5")             // RSA-PKCS1v1.5
	RSA_OAEP           = KeyAlgorithm("RSA-OAEP")           // RSA-OAEP-SHA1
	RSA_OAEP_256       = KeyAlgorithm("RSA-OAEP-256")       // RSA-OAEP-SHA256
	A128KW             = KeyAlgorithm("A128KW")             // AES key wrap (128)
	A192KW             = KeyAlgorithm("A192KW")             // AES key wrap (192)
	A256KW             = KeyAlgorithm("A256KW")             // AES key wrap (256)
	DIRECT             = KeyAlgorithm("dir")                // Direct encryption
	ECDH_ES            = KeyAlgorithm("ECDH-ES")            // ECDH-ES
	ECDH_ES_A128KW     = KeyAlgorithm("ECDH-ES+A128KW")     // ECDH-ES + AES key wrap (128)
	ECDH_ES_A192KW     = KeyAlgorithm("ECDH-ES+A192KW")     // ECDH-ES + AES key wrap (192)
	ECDH_ES_A256KW     = KeyAlgorithm("ECDH-ES+A256KW")     // ECDH-ES + AES key wrap (256)
	A128GCMKW          = KeyAlgorithm("A128GCMKW")          // AES-GCM key wrap (128)
	A192GCMKW          = KeyAlgorithm("A192GCMKW")          // AES-GCM key wrap (192)
	A256GCMKW          = KeyAlgorithm("A256GCMKW")          // AES-GCM key wrap (256)
	PBES2_HS256_A128KW = KeyAlgorithm("PBES2-HS256+A128KW") // PBES2 + HMAC-SHA256 + AES key wrap (128)
	PBES2_HS384_A192KW = KeyAlgorithm("PBES2-HS384+A192KW") // PBES2 + HMAC-SHA384 + AES key wrap (192)
	PBES2_HS512_A256KW = KeyAlgorithm("PBES2-HS512+A256KW") // PBES2 + HMAC-SHA512 + AES key wrap (256)
)

// Signature algorithms
const (
	EdDSA = SignatureAlgorithm("EdDSA")
	HS256 = SignatureAlgorithm("HS256") // HMAC using SHA-256
	HS384 = SignatureAlgorithm("HS384") // HMAC using SHA-384
	HS512 = SignatureAlgorithm("HS512") // HMAC using SHA-512
	RS256 = SignatureAlgorithm("RS256") // RSASSA-PKCS-v1.5 using SHA-256
	RS384 = SignatureAlgorithm("RS384") // RSASSA-PKCS-v1.5 using SHA-384
	RS512 = SignatureAlgorithm("RS512") // RSASSA-PKCS-v1.5 using SHA-512
	ES256 = SignatureAlgorithm("ES256") // ECDSA using P-256 and SHA-256
	ES384 = SignatureAlgorithm("ES384") // ECDSA using P-384 and SHA-384
	ES512 = SignatureAlgorithm("ES512") // ECDSA using P-521 and SHA-512
	PS256 = SignatureAlgorithm("PS256") // RSASSA-PSS using SHA256 and MGF1-SHA256
	PS384 = SignatureAlgorithm("PS384") // RSASSA-PSS using SHA384 and MGF1-SHA384
	PS512 = SignatureAlgorithm("PS512") // RSASSA-PSS using SHA512 and MGF1-SHA512
)

// Content encryption algorithms
const (
	A128CBC_HS256 = ContentEncryption("A128CBC-HS256") // AES-CBC + HMAC-SHA256 (128)
	A192CBC_HS384 = ContentEncryption("A192CBC-HS384") // AES-CBC + HMAC-SHA384 (192)
	A256CBC_HS512 = ContentEncryption("A256CBC-HS512") // AES-CBC + HMAC-SHA512 (256)
	A128GCM       = ContentEncryption("A128GCM")       // AES-GCM (128)
	A192GCM       = ContentEncryption("A192GCM")       // AES-GCM (192)
	A256GCM       = ContentEncryption("A256GCM")       // AES-GCM (256)
)

// Compression algorithms
const (
	NONE    = CompressionAlgorithm("")    // No compression
	DEFLATE = CompressionAlgorithm("DEF") // DEFLATE (RFC 1951)
)

// A key in the protected header of a JWS object. Use of the Header...
// constants is preferred to enhance type safety.
type HeaderKey string

const (
	HeaderType        HeaderKey = "typ" // string
	HeaderContentType           = "cty" // string

	// These are set by go-jose and shouldn't need to be set by consumers of the
	// library.
	headerAlgorithm   = "alg"  // string
	headerEncryption  = "enc"  // ContentEncryption
	headerCompression = "zip"  // CompressionAlgorithm
	headerCritical    = "crit" // []string

	headerAPU = "apu" // *byteBuffer
	headerAPV = "apv" // *byteBuffer
	headerEPK = "epk" // *JSONWebKey
	headerIV  = "iv"  // *byteBuffer
	headerTag = "tag" // *byteBuffer
	headerX5c = "x5c" // []*x509.Certificate

	headerJWK   = "jwk"   // *JSONWebKey
	headerKeyID = "kid"   // string
	headerNonce = "nonce" // string
	headerB64   = "b64"   // bool

	headerP2C = "p2c" // *byteBuffer (int)
	headerP2S = "p2s" // *byteBuffer ([]byte)

)

// supportedCritical is the set of supported extensions that are understood and processed.
var supportedCritical = map[string]bool{
	headerB64: true,
}

// rawHeader represents the JOSE header for JWE/JWS objects (used for parsing).
//
// The decoding of the constituent items is deferred because we want to marshal
// some members into particular structs rather than generic maps, but at the
// same time we need to receive any extra fields unhandled by this library to
// pass through to consuming code in case it wants to examine them.
type rawHeader map[HeaderKey]*json.RawMessage

// Header represents the read-only JOSE header for JWE/JWS objects.
type Header struct {
	KeyID      string
	JSONWebKey *JSONWebKey
	Algorithm  string
	Nonce      string

	// Unverified certificate chain parsed from x5c header.
	certificates []*x509.Certificate

	// Any headers not recognised above get unmarshalled
	// from JSON in a generic manner and placed in this map.
	ExtraHeaders map[HeaderKey]interface{}
}

// Certificates verifies & returns the certificate chain present
// in the x5c header field of a message, if one was present. Returns
// an error if there was no x5c header present or the chain could
// not be validated with the given verify options.
func (h Header) Certificates(opts x509.VerifyOptions) ([][]*x509.Certificate, error) {
	if len(h.certificates) == 0 {
		return nil, errors.New("square/go-jose: no x5c header present in message")
	}

	leaf := h.certificates[0]
	if opts.Intermediates == nil {
		opts.Intermediates = x509.NewCertPool()
		for _, intermediate := range h.certificates[1:] {
			opts.Intermediates.AddCert(intermediate)
		}
	}

	return leaf.Verify(opts)
}

func (parsed rawHeader) set(k HeaderKey, v interface{}) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	parsed[k] = makeRawMessage(b)
	return nil
}

// getString gets a string from the raw JSON, defaulting to "".
func (parsed rawHeader) getString(k HeaderKey) string {
	v, ok := parsed[k]
	if !ok || v == nil {
		return ""
	}
	var s string
	err := json.Unmarshal(*v, &s)
	if err != nil {
		return ""
	}
	return s
}

// getByteBuffer gets a byte buffer from the raw JSON. Returns (nil, nil) if
// not specified.
func (parsed rawHeader) getByteBuffer(k HeaderKey) (*byteBuffer, error) {
	v := parsed[k]
	if v == nil {
		return nil, nil
	}
	var bb *byteBuffer
	err := json.Unmarshal(*v, &bb)
	if err != nil {
		return nil, err
	}
	return bb, nil
}

// getAlgorithm extracts parsed "alg" from the raw JSON as a KeyAlgorithm.
func (parsed rawHeader) getAlgorithm() KeyAlgorithm {
	return KeyAlgorithm(parsed.getString(headerAlgorithm))
}

// getSignatureAlgorithm extracts parsed "alg" from the raw JSON as a SignatureAlgorithm.
func (parsed rawHeader) getSignatureAlgorithm() SignatureAlgorithm {
	return SignatureAlgorithm(parsed.getString(headerAlgorithm))
}

// getEncryption extracts parsed "enc" from the raw JSON.
func (parsed rawHeader) getEncryption() ContentEncryption {
	return ContentEncryption(parsed.getString(headerEncryption))
}

// getCompression extracts parsed "zip" from the raw JSON.
func (parsed rawHeader) getCompression() CompressionAlgorithm {
	return CompressionAlgorithm(parsed.getString(headerCompression))
}

func (parsed rawHeader) getNonce() string {
	return parsed.getString(headerNonce)
}

// getEPK extracts parsed "epk" from the raw JSON.
func (parsed rawHeader) getEPK() (*JSONWebKey, error) {
	v := parsed[headerEPK]
	if v == nil {
		return nil, nil
	}
	var epk *JSONWebKey
	err := json.Unmarshal(*v, &epk)
	if err != nil {
		return nil, err
	}
	return epk, nil
}

// getAPU extracts parsed "apu" from the raw JSON.
func (parsed rawHeader) getAPU() (*byteBuffer, error) {
	return parsed.getByteBuffer(headerAPU)
}

// getAPV extracts parsed "apv" from the raw JSON.
func (parsed rawHeader) getAPV() (*byteBuffer, error) {
	return parsed.getByteBuffer(headerAPV)
}

// getIV extracts parsed "iv" from the raw JSON.
func (parsed rawHeader) getIV() (*byteBuffer, error) {
	return parsed.getByteBuffer(headerIV)
}

// getTag extracts parsed "tag" from the raw JSON.
func (parsed rawHeader) getTag() (*byteBuffer, error) {
	return parsed.getByteBuffer(headerTag)
}

// getJWK extracts parsed "jwk" from the raw JSON.
func (parsed rawHeader) getJWK() (*JSONWebKey, error) {
	v := parsed[headerJWK]
	if v == nil {
		return nil, nil
	}
	var jwk *JSONWebKey
	err := json.Unmarshal(*v, &jwk)
	if err != nil {
		return nil, err
	}
	return jwk, nil
}

// getCritical extracts parsed "crit" from the raw JSON. If omitted, it
// returns an empty slice.
func (parsed rawHeader) getCritical() ([]string, error) {
	v := parsed[headerCritical]
	if v == nil {
		return nil, nil
	}

	var q []string
	err := json.Unmarshal(*v, &q)
	if err != nil {
		return nil, err
	}
	return q, nil
}

// getS2C extracts parsed "p2c" from the raw JSON.
func (parsed rawHeader) getP2C() (int, error) {
	v := parsed[headerP2C]
	if v == nil {
		return 0, nil
	}

	var p2c int
	err := json.Unmarshal(*v, &p2c)
	if err != nil {
		return 0, err
	}
	return p2c, nil
}

// getS2S extracts parsed "p2s" from the raw JSON.
func (parsed rawHeader) getP2S() (*byteBuffer, error) {
	return parsed.getByteBuffer(headerP2S)
}

// getB64 extracts parsed "b64" from the raw JSON, defaulting to true.
func (parsed rawHeader) getB64() (bool, error) {
	v := parsed[headerB64]
	if v == nil {
		return true, nil
	}

	var b64 bool
	err := json.Unmarshal(*v, &b64)
	if err != nil {
		return true, err
	}
	return b64, nil
}

// sanitized produces a cleaned-up header object from the raw JSON.
func (parsed rawHeader) sanitized() (h Header, err error) {
	for k, v := range parsed {
		if v == nil {
			continue
		}
		switch k {
		case headerJWK:
			var jwk *JSONWebKey
			err = json.Unmarshal(*v, &jwk)
			if err != nil {
				err = fmt.Errorf("failed to unmarshal JWK: %v: %#v", err, string(*v))
				return
			}
			h.JSONWebKey = jwk
		case headerKeyID:
			var s string
			err = json.Unmarshal(*v, &s)
			if err != nil {
				err = fmt.Errorf("failed to unmarshal key ID: %v: %#v", err, string(*v))
				return
			}
			h.KeyID = s
		case headerAlgorithm:
			var s string
			err = json.Unmarshal(*v, &s)
			if err != nil {
				err = fmt.Errorf("failed to unmarshal algorithm: %v: %#v", err, string(*v))
				return
			}
			h.Algorithm = s
		case headerNonce:
			var s string
			err = json.Unmarshal(*v, &s)
			if err != nil {
				err = fmt.Errorf("failed to unmarshal nonce: %v: %#v", err, string(*v))
				return
			}
			h.Nonce = s
		case headerX5c:
			c := []string{}
			err = json.Unmarshal(*v, &c)
			if err != nil {
				err = fmt.Errorf("failed to unmarshal x5c header: %v: %#v", err, string(*v))
				return
			}
			h.certificates, err = parseCertificateChain(c)
			if err != nil {
				err = fmt.Errorf("failed to unmarshal x5c header: %v: %#v", err, string(*v))
				return
			}
		default:
			if h.ExtraHeaders == nil {
				h.ExtraHeaders = map[HeaderKey]interface{}{}
			}
			var v2 interface{}
			err = json.Unmarshal(*v, &v2)
			if err != nil {
				err = fmt.Errorf("failed to unmarshal value: %v: %#v", err, string(*v))
				return
			}
			h.ExtraHeaders[k] = v2
		}
	}
	return
}

func parseCertificateChain(chain []string) ([]*x509.Certificate, error) {
	out := make([]*x509.Certificate, len(chain))
	for i, cert := range chain {
		raw, err := base64.StdEncoding.DecodeString(cert)
		if err != nil {
			return nil, err
		}
		out[i], err = x509.ParseCertificate(raw)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (dst rawHeader) isSet(k HeaderKey) bool {
	dvr := dst[k]
	if dvr == nil {
		return false
	}

	var dv interface{}
	err := json.Unmarshal(*dvr, &dv)
	if err != nil {
		return true
	}

	if dvStr, ok := dv.(string); ok {
		return dvStr != ""
	}

	return true
}

// Merge headers from src into dst, giving precedence to headers from l.
func (dst rawHeader) merge(src *rawHeader) {
	if src == nil {
		return
	}

	for k, v := range *src {
		if dst.isSet(k) {
			continue
		}

		dst[k] = v
	}
}

// Get JOSE name of curve
func curveName(crv elliptic.Curve) (string, error) {
	switch crv {
	case elliptic.P256():
		return "P-256", nil
	case elliptic.P384():
		return "P-384", nil
	case elliptic.P521():
		return "P-521", nil
	default:
		return "", fmt.Errorf("square/go-jose: unsupported/unknown elliptic curve")
	}
}

// Get size of curve in bytes
func curveSize(crv elliptic.Curve) int {
	bits := crv.Params().BitSize

	div := bits / 8
	mod := bits % 8

	if mod == 0 {
		return div
	}

	return div + 1
}

func makeRawMessage(b []byte) *json.RawMessage {
	rm := json.RawMessage(b)
	return &rm
}
