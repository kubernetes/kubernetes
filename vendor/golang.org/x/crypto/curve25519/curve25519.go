// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package curve25519 provides an implementation of the X25519 function, which
// performs scalar multiplication on the elliptic curve known as Curve25519.
// See RFC 7748.
//
// This package is a wrapper for the X25519 implementation
// in the crypto/ecdh package.
package curve25519

import "crypto/ecdh"

// ScalarMult sets dst to the product scalar * point.
//
// Deprecated: when provided a low-order point, ScalarMult will set dst to all
// zeroes, irrespective of the scalar. Instead, use the X25519 function, which
// will return an error.
func ScalarMult(dst, scalar, point *[32]byte) {
	if _, err := x25519(dst, scalar[:], point[:]); err != nil {
		// The only error condition for x25519 when the inputs are 32 bytes long
		// is if the output would have been the all-zero value.
		for i := range dst {
			dst[i] = 0
		}
	}
}

// ScalarBaseMult sets dst to the product scalar * base where base is the
// standard generator.
//
// It is recommended to use the X25519 function with Basepoint instead, as
// copying into fixed size arrays can lead to unexpected bugs.
func ScalarBaseMult(dst, scalar *[32]byte) {
	curve := ecdh.X25519()
	priv, err := curve.NewPrivateKey(scalar[:])
	if err != nil {
		panic("curve25519: internal error: scalarBaseMult was not 32 bytes")
	}
	copy(dst[:], priv.PublicKey().Bytes())
}

const (
	// ScalarSize is the size of the scalar input to X25519.
	ScalarSize = 32
	// PointSize is the size of the point input to X25519.
	PointSize = 32
)

// Basepoint is the canonical Curve25519 generator.
var Basepoint []byte

var basePoint = [32]byte{9}

func init() { Basepoint = basePoint[:] }

// X25519 returns the result of the scalar multiplication (scalar * point),
// according to RFC 7748, Section 5. scalar, point and the return value are
// slices of 32 bytes.
//
// scalar can be generated at random, for example with crypto/rand. point should
// be either Basepoint or the output of another X25519 call.
//
// If point is Basepoint (but not if it's a different slice with the same
// contents) a precomputed implementation might be used for performance.
func X25519(scalar, point []byte) ([]byte, error) {
	// Outline the body of function, to let the allocation be inlined in the
	// caller, and possibly avoid escaping to the heap.
	var dst [32]byte
	return x25519(&dst, scalar, point)
}

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
