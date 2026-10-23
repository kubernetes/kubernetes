// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && !fips140v1.0

package ssh

import (
	"crypto"
	"crypto/mldsa"
	"errors"
	"fmt"
)

// mldsaKeyAlgos are the ML-DSA public key algorithms implemented by this
// package.
var mldsaKeyAlgos = []string{
	KeyAlgoMLDSA44,
	KeyAlgoMLDSA65,
	KeyAlgoMLDSA87,
}

// mldsaCertAlgos are the certificate algorithms for the keys in mldsaKeyAlgos.
var mldsaCertAlgos = []string{
	CertAlgoMLDSA44v01Go,
	CertAlgoMLDSA65v01Go,
	CertAlgoMLDSA87v01Go,
}

var errNotMLDSAKey = errors.New("ssh: not an ML-DSA key")

// mldsaParameters returns the ML-DSA parameter set for the given public key
// algorithm name.
func mldsaParameters(algo string) (mldsa.Parameters, error) {
	switch algo {
	case KeyAlgoMLDSA44:
		return mldsa.MLDSA44(), nil
	case KeyAlgoMLDSA65:
		return mldsa.MLDSA65(), nil
	case KeyAlgoMLDSA87:
		return mldsa.MLDSA87(), nil
	default:
		return mldsa.Parameters{}, fmt.Errorf("ssh: not an ML-DSA algorithm: %q", algo)
	}
}

// mldsaKeyAlgo returns the public key algorithm name for the given ML-DSA
// parameter set.
func mldsaKeyAlgo(params mldsa.Parameters) string {
	switch params {
	case mldsa.MLDSA44():
		return KeyAlgoMLDSA44
	case mldsa.MLDSA65():
		return KeyAlgoMLDSA65
	case mldsa.MLDSA87():
		return KeyAlgoMLDSA87
	default:
		panic("ssh: internal error: unknown ML-DSA parameter set: " + params.String())
	}
}

// mldsaPublicKey implements PublicKey for the ML-DSA algorithms of [SSH-MLDSA].
type mldsaPublicKey struct {
	key *mldsa.PublicKey
}

func (k *mldsaPublicKey) Type() string {
	return mldsaKeyAlgo(k.key.Parameters())
}

func parseMLDSA(in []byte, algo string) (out PublicKey, rest []byte, err error) {
	params, err := mldsaParameters(algo)
	if err != nil {
		return nil, nil, err
	}

	var w struct {
		KeyBytes []byte
		Rest     []byte `ssh:"rest"`
	}

	if err := Unmarshal(in, &w); err != nil {
		return nil, nil, err
	}

	key, err := mldsa.NewPublicKey(params, w.KeyBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("ssh: invalid %s public key: %v", params, err)
	}

	return &mldsaPublicKey{key: key}, w.Rest, nil
}

func (k *mldsaPublicKey) Marshal() []byte {
	w := struct {
		Name     string
		KeyBytes []byte
	}{
		k.Type(),
		k.key.Bytes(),
	}
	return Marshal(&w)
}

func (k *mldsaPublicKey) Verify(b []byte, sig *Signature) error {
	if sig.Format != k.Type() {
		return fmt.Errorf("ssh: signature type %s for key type %s", sig.Format, k.Type())
	}
	// nil options select pure ML-DSA with an empty context, as required by
	// [SSH-MLDSA], Sections 6 and 8.
	if err := mldsa.Verify(k.key, b, sig.Blob, nil); err != nil {
		return errors.New("ssh: signature did not verify")
	}

	return nil
}

func (k *mldsaPublicKey) CryptoPublicKey() crypto.PublicKey {
	return k.key
}

// newMLDSAPublicKey returns a PublicKey wrapping key.
func newMLDSAPublicKey(key crypto.PublicKey) (PublicKey, error) {
	k, ok := key.(*mldsa.PublicKey)
	if !ok {
		return nil, errNotMLDSAKey
	}
	if k == nil || *k == (mldsa.PublicKey{}) {
		return nil, errors.New("ssh: invalid ML-DSA public key")
	}
	return &mldsaPublicKey{key: k}, nil
}
