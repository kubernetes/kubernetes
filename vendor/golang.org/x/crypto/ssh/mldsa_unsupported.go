// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.27 || fips140v1.0

package ssh

import (
	"crypto"
	"errors"
	"fmt"
)

var mldsaKeyAlgos, mldsaCertAlgos []string

var errMLDSAUnsupported = fmt.Errorf("ssh: ML-DSA is not available in this build: %w",
	errors.ErrUnsupported)

var errNotMLDSAKey = errors.New("ssh: not an ML-DSA key")

func parseMLDSA(in []byte, algo string) (out PublicKey, rest []byte, err error) {
	return nil, nil, errMLDSAUnsupported
}

func newMLDSAPublicKey(key crypto.PublicKey) (PublicKey, error) {
	// In this version of Go, *mldsa.PublicKey does not exist.
	return nil, errNotMLDSAKey
}
