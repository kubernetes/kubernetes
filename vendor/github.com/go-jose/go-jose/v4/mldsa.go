//go:build go1.27

/*-
 * Copyright 2026 The Go JOSE Authors.
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
	"crypto/mldsa"
	"errors"
	"fmt"
)

// mldsaParamsFor maps a JOSE signature algorithm to its ML-DSA parameter set.
//
// See RFC 9964, which registers ML-DSA-44, ML-DSA-65 and ML-DSA-87 as JWS "alg"
// values for the FIPS 204 parameter sets of the same names.
func mldsaParamsFor(alg SignatureAlgorithm) (mldsa.Parameters, bool) {
	switch alg {
	case ML_DSA_44:
		return mldsa.MLDSA44(), true
	case ML_DSA_65:
		return mldsa.MLDSA65(), true
	case ML_DSA_87:
		return mldsa.MLDSA87(), true
	default:
		return mldsa.Parameters{}, false
	}
}

// mldsaAlgFor maps an ML-DSA parameter set to its JOSE signature algorithm. An
// AKP JWK carries no "crv", so the parameter set is the only thing that
// identifies which algorithm a key belongs to.
func mldsaAlgFor(params mldsa.Parameters) (SignatureAlgorithm, bool) {
	switch params {
	case mldsa.MLDSA44():
		return ML_DSA_44, true
	case mldsa.MLDSA65():
		return ML_DSA_65, true
	case mldsa.MLDSA87():
		return ML_DSA_87, true
	default:
		return "", false
	}
}

// mldsaPublicOK reports whether pub is usable. crypto/mldsa's constructors
// validate key material, but PublicKey is an exported struct with unexported
// fields, so a caller can bypass them with a composite literal. Parameters()
// panics on such a zero value, so every hook must screen for it.
func mldsaPublicOK(pub *mldsa.PublicKey) bool {
	return pub != nil && *pub != (mldsa.PublicKey{})
}

// mldsaPrivateOK reports whether priv is usable. See mldsaPublicOK.
func mldsaPrivateOK(priv *mldsa.PrivateKey) bool {
	return priv != nil && *priv != (mldsa.PrivateKey{})
}

// mldsaKeyInfo reports whether key is a usable ML-DSA key, and if so whether it
// is the public half. "Usable" excludes both a nil key and a zero-value key
// constructed by a caller bypassing crypto/mldsa's validating constructors; see
// mldsaPublicOK.
func mldsaKeyInfo(key interface{}) (isPublic bool, ok bool) {
	switch k := key.(type) {
	case *mldsa.PublicKey:
		return true, mldsaPublicOK(k)
	case *mldsa.PrivateKey:
		return false, mldsaPrivateOK(k)
	default:
		return false, false
	}
}

// mldsaPublicOf returns the public half of an ML-DSA private key.
func mldsaPublicOf(key interface{}) (interface{}, bool) {
	if k, ok := key.(*mldsa.PrivateKey); ok && mldsaPrivateOK(k) {
		return k.PublicKey(), true
	}
	return nil, false
}

// mldsaPrivateKeySigner signs payloads with an ML-DSA private key. The algorithm
// is resolved from the key's parameter set once at construction and cached.
type mldsaPrivateKeySigner struct {
	privateKey *mldsa.PrivateKey
	alg        SignatureAlgorithm
}

// mldsaPublicKeyVerifier verifies payloads with an ML-DSA public key.
type mldsaPublicKeyVerifier struct {
	publicKey *mldsa.PublicKey
	alg       SignatureAlgorithm
}

// mldsaSigner creates a recipientSigInfo for an ML-DSA private key. The second
// return value reports whether key was an ML-DSA key at all.
func mldsaSigner(sigAlg SignatureAlgorithm, key interface{}) (recipientSigInfo, bool, error) {
	privateKey, ok := key.(*mldsa.PrivateKey)
	if !ok {
		return recipientSigInfo{}, false, nil
	}
	if !mldsaPrivateOK(privateKey) {
		return recipientSigInfo{}, true, errors.New("invalid private key")
	}

	alg, ok := mldsaAlgFor(privateKey.PublicKey().Parameters())
	if !ok || alg != sigAlg {
		return recipientSigInfo{}, true, ErrUnsupportedAlgorithm
	}

	return recipientSigInfo{
		sigAlg: sigAlg,
		publicKey: staticPublicKey(&JSONWebKey{
			Key: privateKey.PublicKey(),
		}),
		signer: &mldsaPrivateKeySigner{
			privateKey: privateKey,
			alg:        alg,
		},
	}, true, nil
}

// mldsaVerifier creates a payloadVerifier for an ML-DSA public key.
func mldsaVerifier(key interface{}) (payloadVerifier, bool, error) {
	publicKey, ok := key.(*mldsa.PublicKey)
	if !ok {
		return nil, false, nil
	}
	if !mldsaPublicOK(publicKey) {
		return nil, true, errors.New("invalid public key")
	}

	alg, ok := mldsaAlgFor(publicKey.Parameters())
	if !ok {
		return nil, true, ErrUnsupportedKeyType
	}

	return &mldsaPublicKeyVerifier{
		publicKey: publicKey,
		alg:       alg,
	}, true, nil
}

// mldsaRawJWK converts an ML-DSA key to its AKP JWK representation. The "alg"
// member is derived from the key's own parameter set, because RFC 9964 requires
// it on every AKP key and it is the only member identifying the parameter set.
func mldsaRawJWK(key interface{}) (*rawJSONWebKey, bool, error) {
	var (
		pub  *mldsa.PublicKey
		priv *mldsa.PrivateKey
	)

	switch k := key.(type) {
	case *mldsa.PublicKey:
		if !mldsaPublicOK(k) {
			return nil, true, errors.New("go-jose/go-jose: invalid public key")
		}
		pub = k
	case *mldsa.PrivateKey:
		if !mldsaPrivateOK(k) {
			return nil, true, errors.New("go-jose/go-jose: invalid private key")
		}
		priv, pub = k, k.PublicKey()
	default:
		return nil, false, nil
	}

	alg, ok := mldsaAlgFor(pub.Parameters())
	if !ok {
		return nil, true, ErrUnsupportedKeyType
	}

	raw := &rawJSONWebKey{
		Kty: "AKP",
		Alg: string(alg),
		Pub: newBuffer(pub.Bytes()),
	}
	if priv != nil {
		// RFC 9964: priv MUST be the seed, and MUST be 32 bytes. Bytes() returns
		// exactly that.
		raw.Priv = newBuffer(priv.Bytes())
	}

	return raw, true, nil
}

// mldsaParseJWK builds an ML-DSA key from an AKP JWK.
//
// RFC 9964 makes "alg" REQUIRED for AKP keys: unlike EC and OKP there is no
// "crv", so "alg" is the only member identifying the parameter set. A JWK
// without a recognized "alg" is therefore not merely unusual but unusable, and
// is rejected with ErrUnsupportedKeyType.
//
// A parsed key can never carry an alg/material disagreement: params is derived
// from raw.Alg below, and mldsa.NewPrivateKey/NewPublicKey both reject key
// material whose length disagrees with those params.
//
// When certPub is non-nil it is checked against the key's public half here,
// using (*mldsa.PublicKey).Equal, so that the caller does not have to compare
// unexported crypto/mldsa internals with reflect.DeepEqual.
func mldsaParseJWK(raw *rawJSONWebKey, certPub interface{}) (interface{}, error) {
	params, ok := mldsaParamsFor(SignatureAlgorithm(raw.Alg))
	if !ok {
		return nil, ErrUnsupportedKeyType
	}

	// RFC 9964 makes "pub" REQUIRED on every AKP JWK.
	if raw.Pub == nil {
		return nil, ErrUnsupportedKeyType
	}

	var (
		key interface{}
		pub *mldsa.PublicKey
	)

	if raw.Priv != nil {
		// RFC 9964: priv MUST be the seed, and MUST be 32 bytes. NewPrivateKey
		// enforces the length.
		privateKey, err := mldsa.NewPrivateKey(params, raw.Priv.bytes())
		if err != nil {
			return nil, fmt.Errorf("go-jose/go-jose: invalid AKP private key: %w", err)
		}
		pub = privateKey.PublicKey()
		if !bytes.Equal(raw.Pub.bytes(), pub.Bytes()) {
			return nil, errors.New("go-jose/go-jose: invalid AKP key, pub does not match the public key derived from priv")
		}
		key = privateKey
	} else {
		// NewPublicKey rejects an encoding whose length disagrees with params,
		// which is what catches an alg/pub mismatch.
		publicKey, err := mldsa.NewPublicKey(params, raw.Pub.bytes())
		if err != nil {
			return nil, fmt.Errorf("go-jose/go-jose: invalid AKP public key: %w", err)
		}
		pub = publicKey
		key = publicKey
	}

	if certPub != nil && !pub.Equal(certPub) {
		return nil, errors.New("go-jose/go-jose: invalid JWK, public keys in key and x5c fields do not match")
	}

	return key, nil
}

func (ctx mldsaPrivateKeySigner) signPayload(payload []byte, alg SignatureAlgorithm) (Signature, error) {
	if alg != ctx.alg {
		return Signature{}, ErrUnsupportedAlgorithm
	}

	// crypto.Hash(0) selects pure ML-DSA with an empty context string, which is
	// what RFC 9964 specifies for JOSE. Note that crypto/mldsa ignores the
	// io.Reader and draws its own randomness.
	sig, err := ctx.privateKey.Sign(randReader, payload, crypto.Hash(0))
	if err != nil {
		return Signature{}, err
	}

	return Signature{
		Signature: sig,
		protected: &rawHeader{},
	}, nil
}

func (ctx mldsaPublicKeyVerifier) verifyPayload(payload []byte, signature []byte, alg SignatureAlgorithm) error {
	// Reject a header algorithm that disagrees with the key's parameter set, so
	// an ML-DSA-44 key can never be used under an ML-DSA-87 header.
	if alg != ctx.alg {
		return ErrUnsupportedAlgorithm
	}

	// A nil Options means an empty context string, per RFC 9964.
	if err := mldsa.Verify(ctx.publicKey, payload, signature, nil); err != nil {
		return fmt.Errorf("go-jose/go-jose: %s signature failed to verify: %w", ctx.alg, err)
	}

	return nil
}

// mldsaThumbprintTemplate is the JWK Thumbprint input for AKP keys. RFC 9964
// gives the required members as "alg", "kty" and "pub"; RFC 7638 requires them
// in lexicographic order with no whitespace. Note that AKP is the only key type
// in this package whose thumbprint includes "alg".
const mldsaThumbprintTemplate = `{"alg":"%s","kty":"AKP","pub":"%s"}`

// mldsaThumbprintInput builds the JWK Thumbprint input for an ML-DSA key. A
// private key thumbprints as its public half.
func mldsaThumbprintInput(key interface{}) (string, bool, error) {
	var pub *mldsa.PublicKey

	switch k := key.(type) {
	case *mldsa.PublicKey:
		if mldsaPublicOK(k) {
			pub = k
		}
	case *mldsa.PrivateKey:
		if mldsaPrivateOK(k) {
			pub = k.PublicKey()
		}
	default:
		return "", false, nil
	}

	if pub == nil {
		return "", true, errors.New("go-jose/go-jose: invalid ML-DSA key")
	}

	alg, ok := mldsaAlgFor(pub.Parameters())
	if !ok {
		return "", true, ErrUnsupportedKeyType
	}

	return fmt.Sprintf(mldsaThumbprintTemplate, alg, newBuffer(pub.Bytes()).base64()), true, nil
}
