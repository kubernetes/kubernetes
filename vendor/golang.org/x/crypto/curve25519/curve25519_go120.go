// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20

package curve25519

import "crypto/ecdh"

func x25519(dst *[32]byte, scalar, point []byte) ([]byte, error) {
	curve := ecdh.X25519()
	pub, err := curve.NewPublicKey(point)
	if err != nil {
		return nil, err
	}
	priv, err := curve.NewPrivateKey(scalar)
	if err != nil {
		return nil, err
	}
	out, err := priv.ECDH(pub)
	if err != nil {
		return nil, err
	}
	copy(dst[:], out)
	return dst[:], nil
}

func scalarMult(dst, scalar, point *[32]byte) {
	if _, err := x25519(dst, scalar[:], point[:]); err != nil {
		// The only error condition for x25519 when the inputs are 32 bytes long
		// is if the output would have been the all-zero value.
		for i := range dst {
			dst[i] = 0
		}
	}
}

func scalarBaseMult(dst, scalar *[32]byte) {
	curve := ecdh.X25519()
	priv, err := curve.NewPrivateKey(scalar[:])
	if err != nil {
		panic("curve25519: internal error: scalarBaseMult was not 32 bytes")
	}
	copy(dst[:], priv.PublicKey().Bytes())
}
