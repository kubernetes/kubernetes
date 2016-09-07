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
	"errors"
	"fmt"
)

// KeyAlgorithm represents a key management algorithm.
type KeyAlgorithm string

// SignatureAlgorithm represents a signature (or MAC) algorithm.
type SignatureAlgorithm string

// ContentEncryption represents a content encryption algorithm.
type ContentEncryption string

// CompressionAlgorithm represents an algorithm used for plaintext compression.
type CompressionAlgorithm string

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

// rawHeader represents the JOSE header for JWE/JWS objects (used for parsing).
type rawHeader struct {
	Alg   string               `json:"alg,omitempty"`
	Enc   ContentEncryption    `json:"enc,omitempty"`
	Zip   CompressionAlgorithm `json:"zip,omitempty"`
	Crit  []string             `json:"crit,omitempty"`
	Apu   *byteBuffer          `json:"apu,omitempty"`
	Apv   *byteBuffer          `json:"apv,omitempty"`
	Epk   *JsonWebKey          `json:"epk,omitempty"`
	Iv    *byteBuffer          `json:"iv,omitempty"`
	Tag   *byteBuffer          `json:"tag,omitempty"`
	Jwk   *JsonWebKey          `json:"jwk,omitempty"`
	Kid   string               `json:"kid,omitempty"`
	Nonce string               `json:"nonce,omitempty"`
}

// JoseHeader represents the read-only JOSE header for JWE/JWS objects.
type JoseHeader struct {
	KeyID      string
	JsonWebKey *JsonWebKey
	Algorithm  string
	Nonce      string
}

// sanitized produces a cleaned-up header object from the raw JSON.
func (parsed rawHeader) sanitized() JoseHeader {
	return JoseHeader{
		KeyID:      parsed.Kid,
		JsonWebKey: parsed.Jwk,
		Algorithm:  parsed.Alg,
		Nonce:      parsed.Nonce,
	}
}

// Merge headers from src into dst, giving precedence to headers from l.
func (dst *rawHeader) merge(src *rawHeader) {
	if src == nil {
		return
	}

	if dst.Alg == "" {
		dst.Alg = src.Alg
	}
	if dst.Enc == "" {
		dst.Enc = src.Enc
	}
	if dst.Zip == "" {
		dst.Zip = src.Zip
	}
	if dst.Crit == nil {
		dst.Crit = src.Crit
	}
	if dst.Crit == nil {
		dst.Crit = src.Crit
	}
	if dst.Apu == nil {
		dst.Apu = src.Apu
	}
	if dst.Apv == nil {
		dst.Apv = src.Apv
	}
	if dst.Epk == nil {
		dst.Epk = src.Epk
	}
	if dst.Iv == nil {
		dst.Iv = src.Iv
	}
	if dst.Tag == nil {
		dst.Tag = src.Tag
	}
	if dst.Kid == "" {
		dst.Kid = src.Kid
	}
	if dst.Jwk == nil {
		dst.Jwk = src.Jwk
	}
	if dst.Nonce == "" {
		dst.Nonce = src.Nonce
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
