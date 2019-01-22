// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tls

import (
	"crypto"
	"crypto/dsa"
	"crypto/ecdsa"
	_ "crypto/md5" // For registration side-effect
	"crypto/rand"
	"crypto/rsa"
	_ "crypto/sha1"   // For registration side-effect
	_ "crypto/sha256" // For registration side-effect
	_ "crypto/sha512" // For registration side-effect
	"errors"
	"fmt"
	"log"
	"math/big"

	"github.com/google/certificate-transparency-go/asn1"
)

type dsaSig struct {
	R, S *big.Int
}

func generateHash(algo HashAlgorithm, data []byte) ([]byte, crypto.Hash, error) {
	var hashType crypto.Hash
	switch algo {
	case MD5:
		hashType = crypto.MD5
	case SHA1:
		hashType = crypto.SHA1
	case SHA224:
		hashType = crypto.SHA224
	case SHA256:
		hashType = crypto.SHA256
	case SHA384:
		hashType = crypto.SHA384
	case SHA512:
		hashType = crypto.SHA512
	default:
		return nil, hashType, fmt.Errorf("unsupported Algorithm.Hash in signature: %v", algo)
	}

	hasher := hashType.New()
	if _, err := hasher.Write(data); err != nil {
		return nil, hashType, fmt.Errorf("failed to write to hasher: %v", err)
	}
	return hasher.Sum([]byte{}), hashType, nil
}

// VerifySignature verifies that the passed in signature over data was created by the given PublicKey.
func VerifySignature(pubKey crypto.PublicKey, data []byte, sig DigitallySigned) error {
	hash, hashType, err := generateHash(sig.Algorithm.Hash, data)
	if err != nil {
		return err
	}

	switch sig.Algorithm.Signature {
	case RSA:
		rsaKey, ok := pubKey.(*rsa.PublicKey)
		if !ok {
			return fmt.Errorf("cannot verify RSA signature with %T key", pubKey)
		}
		if err := rsa.VerifyPKCS1v15(rsaKey, hashType, hash, sig.Signature); err != nil {
			return fmt.Errorf("failed to verify rsa signature: %v", err)
		}
	case DSA:
		dsaKey, ok := pubKey.(*dsa.PublicKey)
		if !ok {
			return fmt.Errorf("cannot verify DSA signature with %T key", pubKey)
		}
		var dsaSig dsaSig
		rest, err := asn1.Unmarshal(sig.Signature, &dsaSig)
		if err != nil {
			return fmt.Errorf("failed to unmarshal DSA signature: %v", err)
		}
		if len(rest) != 0 {
			log.Printf("Garbage following signature %v", rest)
		}
		if dsaSig.R.Sign() <= 0 || dsaSig.S.Sign() <= 0 {
			return errors.New("DSA signature contained zero or negative values")
		}
		if !dsa.Verify(dsaKey, hash, dsaSig.R, dsaSig.S) {
			return errors.New("failed to verify DSA signature")
		}
	case ECDSA:
		ecdsaKey, ok := pubKey.(*ecdsa.PublicKey)
		if !ok {
			return fmt.Errorf("cannot verify ECDSA signature with %T key", pubKey)
		}
		var ecdsaSig dsaSig
		rest, err := asn1.Unmarshal(sig.Signature, &ecdsaSig)
		if err != nil {
			return fmt.Errorf("failed to unmarshal ECDSA signature: %v", err)
		}
		if len(rest) != 0 {
			log.Printf("Garbage following signature %v", rest)
		}
		if ecdsaSig.R.Sign() <= 0 || ecdsaSig.S.Sign() <= 0 {
			return errors.New("ECDSA signature contained zero or negative values")
		}

		if !ecdsa.Verify(ecdsaKey, hash, ecdsaSig.R, ecdsaSig.S) {
			return errors.New("failed to verify ECDSA signature")
		}
	default:
		return fmt.Errorf("unsupported Algorithm.Signature in signature: %v", sig.Algorithm.Hash)
	}
	return nil
}

// CreateSignature builds a signature over the given data using the specified hash algorithm and private key.
func CreateSignature(privKey crypto.PrivateKey, hashAlgo HashAlgorithm, data []byte) (DigitallySigned, error) {
	var sig DigitallySigned
	sig.Algorithm.Hash = hashAlgo
	hash, hashType, err := generateHash(sig.Algorithm.Hash, data)
	if err != nil {
		return sig, err
	}

	switch privKey := privKey.(type) {
	case rsa.PrivateKey:
		sig.Algorithm.Signature = RSA
		sig.Signature, err = rsa.SignPKCS1v15(rand.Reader, &privKey, hashType, hash)
		return sig, err
	case ecdsa.PrivateKey:
		sig.Algorithm.Signature = ECDSA
		var ecdsaSig dsaSig
		ecdsaSig.R, ecdsaSig.S, err = ecdsa.Sign(rand.Reader, &privKey, hash)
		if err != nil {
			return sig, err
		}
		sig.Signature, err = asn1.Marshal(ecdsaSig)
		return sig, err
	default:
		return sig, fmt.Errorf("unsupported private key type %T", privKey)
	}
}
