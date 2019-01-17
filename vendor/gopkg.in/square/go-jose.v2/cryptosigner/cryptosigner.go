/*-
 * Copyright 2018 Square Inc.
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

// Package cryptosigner implements an OpaqueSigner that wraps a "crypto".Signer
//
// https://godoc.org/crypto#Signer
package cryptosigner

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/rsa"
	"encoding/asn1"
	"io"
	"math/big"

	"golang.org/x/crypto/ed25519"
	"gopkg.in/square/go-jose.v2"
)

// Opaque creates an OpaqueSigner from a "crypto".Signer
func Opaque(s crypto.Signer) jose.OpaqueSigner {
	pk := &jose.JSONWebKey{
		Key: s.Public(),
	}
	return &cryptoSigner{signer: s, rand: rand.Reader, pk: pk}
}

type cryptoSigner struct {
	pk     *jose.JSONWebKey
	signer crypto.Signer
	rand   io.Reader
}

func (s *cryptoSigner) Public() *jose.JSONWebKey {
	return s.pk
}

func (s *cryptoSigner) Algs() []jose.SignatureAlgorithm {
	switch s.signer.Public().(type) {
	case ed25519.PublicKey:
		return []jose.SignatureAlgorithm{jose.EdDSA}
	case *ecdsa.PublicKey:
		// This could be more precise
		return []jose.SignatureAlgorithm{jose.ES256, jose.ES384, jose.ES512}
	case *rsa.PublicKey:
		return []jose.SignatureAlgorithm{jose.RS256, jose.RS384, jose.RS512, jose.PS256, jose.PS384, jose.PS512}
	default:
		return nil
	}
}

func (s *cryptoSigner) SignPayload(payload []byte, alg jose.SignatureAlgorithm) ([]byte, error) {
	var hash crypto.Hash
	switch alg {
	case jose.EdDSA:
	case jose.RS256, jose.PS256, jose.ES256:
		hash = crypto.SHA256
	case jose.RS384, jose.PS384, jose.ES384:
		hash = crypto.SHA384
	case jose.RS512, jose.PS512, jose.ES512:
		hash = crypto.SHA512
	default:
		return nil, jose.ErrUnsupportedAlgorithm
	}

	var hashed []byte
	if hash != crypto.Hash(0) {
		hasher := hash.New()
		if _, err := hasher.Write(payload); err != nil {
			return nil, err
		}
		hashed = hasher.Sum(nil)
	}

	var (
		out []byte
		err error
	)
	switch alg {
	case jose.EdDSA:
		out, err = s.signer.Sign(s.rand, payload, crypto.Hash(0))
	case jose.ES256, jose.ES384, jose.ES512:
		var byteLen int
		switch alg {
		case jose.ES256:
			byteLen = 32
		case jose.ES384:
			byteLen = 48
		case jose.ES512:
			byteLen = 66
		}
		var b []byte
		b, err = s.signer.Sign(s.rand, hashed, hash)
		if err != nil {
			return nil, err
		}

		sig := struct {
			R, S *big.Int
		}{}
		if _, err = asn1.Unmarshal(b, &sig); err != nil {
			return nil, err
		}

		rBytes := sig.R.Bytes()
		rBytesPadded := make([]byte, byteLen)
		copy(rBytesPadded[byteLen-len(rBytes):], rBytes)

		sBytes := sig.S.Bytes()
		sBytesPadded := make([]byte, byteLen)
		copy(sBytesPadded[byteLen-len(sBytes):], sBytes)

		out = append(rBytesPadded, sBytesPadded...)
	case jose.RS256, jose.RS384, jose.RS512:
		out, err = s.signer.Sign(s.rand, hashed, hash)
	case jose.PS256, jose.PS384, jose.PS512:
		out, err = s.signer.Sign(s.rand, hashed, &rsa.PSSOptions{
			SaltLength: rsa.PSSSaltLengthAuto,
			Hash:       hash,
		})
	}
	return out, err
}
