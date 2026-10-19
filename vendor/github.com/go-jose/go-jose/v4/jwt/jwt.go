/*-
 * Copyright 2016 Zbigniew Mandziejewicz
 * Copyright 2016 Square, Inc.
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

package jwt

import (
	"fmt"
	"strings"

	jose "github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/json"
)

// JSONWebToken represents a JSON Web Token (as specified in RFC7519).
type JSONWebToken struct {
	payload           func(k interface{}) ([]byte, error)
	unverifiedPayload func() []byte
	Headers           []jose.Header
}

type NestedJSONWebToken struct {
	enc     *jose.JSONWebEncryption
	Headers []jose.Header
	// Used when parsing and decrypting an input
	allowedSignatureAlgorithms []jose.SignatureAlgorithm
}

// Claims deserializes a JSONWebToken into dest using the provided key.
func (t *JSONWebToken) Claims(key interface{}, dest ...interface{}) error {
	b, err := t.payload(key)
	if err != nil {
		return err
	}

	for _, d := range dest {
		if err := json.Unmarshal(b, d); err != nil {
			return err
		}
	}

	return nil
}

// UnsafeClaimsWithoutVerification deserializes the claims of a
// JSONWebToken into the dests. For signed JWTs, the claims are not
// verified. This function won't work for encrypted JWTs.
func (t *JSONWebToken) UnsafeClaimsWithoutVerification(dest ...interface{}) error {
	if t.unverifiedPayload == nil {
		return fmt.Errorf("go-jose/go-jose: Cannot get unverified claims")
	}
	claims := t.unverifiedPayload()
	for _, d := range dest {
		if err := json.Unmarshal(claims, d); err != nil {
			return err
		}
	}
	return nil
}

func (t *NestedJSONWebToken) Decrypt(decryptionKey interface{}) (*JSONWebToken, error) {
	b, err := t.enc.Decrypt(decryptionKey)
	if err != nil {
		return nil, err
	}

	sig, err := ParseSigned(string(b), t.allowedSignatureAlgorithms)
	if err != nil {
		return nil, err
	}

	return sig, nil
}

// ParseSigned parses token from JWS form.
func ParseSigned(s string, signatureAlgorithms []jose.SignatureAlgorithm) (*JSONWebToken, error) {
	sig, err := jose.ParseSignedCompact(s, signatureAlgorithms)
	if err != nil {
		return nil, err
	}
	headers := make([]jose.Header, len(sig.Signatures))
	for i, signature := range sig.Signatures {
		headers[i] = signature.Header
	}

	return &JSONWebToken{
		payload:           sig.Verify,
		unverifiedPayload: sig.UnsafePayloadWithoutVerification,
		Headers:           headers,
	}, nil
}

func validateKeyEncryptionAlgorithm(algs []jose.KeyAlgorithm) error {
	for _, alg := range algs {
		switch alg {
		case jose.ED25519,
			jose.RSA1_5,
			jose.RSA_OAEP,
			jose.RSA_OAEP_256,
			jose.ECDH_ES,
			jose.ECDH_ES_A128KW,
			jose.ECDH_ES_A192KW,
			jose.ECDH_ES_A256KW:
			return fmt.Errorf("asymmetric encryption algorithms not supported for JWT: "+
				"invalid key encryption algorithm: %s", alg)
		case jose.PBES2_HS256_A128KW,
			jose.PBES2_HS384_A192KW,
			jose.PBES2_HS512_A256KW:
			return fmt.Errorf("password-based encryption not supported for JWT: "+
				"invalid key encryption algorithm: %s", alg)
		}
	}
	return nil
}

func parseEncryptedCompact(
	s string,
	keyAlgorithms []jose.KeyAlgorithm,
	contentEncryption []jose.ContentEncryption,
) (*jose.JSONWebEncryption, error) {
	err := validateKeyEncryptionAlgorithm(keyAlgorithms)
	if err != nil {
		return nil, err
	}
	enc, err := jose.ParseEncryptedCompact(s, keyAlgorithms, contentEncryption)
	if err != nil {
		return nil, err
	}
	return enc, nil
}

// ParseEncrypted parses token from JWE form.
//
// The keyAlgorithms and contentEncryption parameters are used to validate the "alg" and "enc"
// header parameters respectively. They must be nonempty, and each "alg" or "enc" header in
// parsed data must contain a value that is present in the corresponding parameter. That
// includes the protected and unprotected headers as well as all recipients. To accept
// multiple algorithms, pass a slice of all the algorithms you want to accept.
func ParseEncrypted(s string,
	keyAlgorithms []jose.KeyAlgorithm,
	contentEncryption []jose.ContentEncryption,
) (*JSONWebToken, error) {
	enc, err := parseEncryptedCompact(s, keyAlgorithms, contentEncryption)
	if err != nil {
		return nil, err
	}

	return &JSONWebToken{
		payload: enc.Decrypt,
		Headers: []jose.Header{enc.Header},
	}, nil
}

// ParseSignedAndEncrypted parses signed-then-encrypted token from JWE form.
//
// The encryptionKeyAlgorithms and contentEncryption parameters are used to validate the "alg" and "enc"
// header parameters, respectively, of the outer JWE. They must be nonempty, and each "alg" or "enc"
// header in parsed data must contain a value that is present in the corresponding parameter. That
// includes the protected and unprotected headers as well as all recipients. To accept
// multiple algorithms, pass a slice of all the algorithms you want to accept.
//
// The signatureAlgorithms parameter is used to validate the "alg" header parameter of the
// inner JWS. It must be nonempty, and the "alg" header in the inner JWS must contain a value
// that is present in the parameter.
func ParseSignedAndEncrypted(s string,
	encryptionKeyAlgorithms []jose.KeyAlgorithm,
	contentEncryption []jose.ContentEncryption,
	signatureAlgorithms []jose.SignatureAlgorithm,
) (*NestedJSONWebToken, error) {
	enc, err := parseEncryptedCompact(s, encryptionKeyAlgorithms, contentEncryption)
	if err != nil {
		return nil, err
	}

	contentType, _ := enc.Header.ExtraHeaders[jose.HeaderContentType].(string)
	if strings.ToUpper(contentType) != "JWT" {
		return nil, ErrInvalidContentType
	}

	return &NestedJSONWebToken{
		allowedSignatureAlgorithms: signatureAlgorithms,
		enc:                        enc,
		Headers:                    []jose.Header{enc.Header},
	}, nil
}
