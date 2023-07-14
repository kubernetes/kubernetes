// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.20

package curve25519

import (
	"crypto/subtle"
	"errors"
	"strconv"

	"golang.org/x/crypto/curve25519/internal/field"
)

func scalarMult(dst, scalar, point *[32]byte) {
	var e [32]byte

	copy(e[:], scalar[:])
	e[0] &= 248
	e[31] &= 127
	e[31] |= 64

	var x1, x2, z2, x3, z3, tmp0, tmp1 field.Element
	x1.SetBytes(point[:])
	x2.One()
	x3.Set(&x1)
	z3.One()

	swap := 0
	for pos := 254; pos >= 0; pos-- {
		b := e[pos/8] >> uint(pos&7)
		b &= 1
		swap ^= int(b)
		x2.Swap(&x3, swap)
		z2.Swap(&z3, swap)
		swap = int(b)

		tmp0.Subtract(&x3, &z3)
		tmp1.Subtract(&x2, &z2)
		x2.Add(&x2, &z2)
		z2.Add(&x3, &z3)
		z3.Multiply(&tmp0, &x2)
		z2.Multiply(&z2, &tmp1)
		tmp0.Square(&tmp1)
		tmp1.Square(&x2)
		x3.Add(&z3, &z2)
		z2.Subtract(&z3, &z2)
		x2.Multiply(&tmp1, &tmp0)
		tmp1.Subtract(&tmp1, &tmp0)
		z2.Square(&z2)

		z3.Mult32(&tmp1, 121666)
		x3.Square(&x3)
		tmp0.Add(&tmp0, &z3)
		z3.Multiply(&x1, &z2)
		z2.Multiply(&tmp1, &tmp0)
	}

	x2.Swap(&x3, swap)
	z2.Swap(&z3, swap)

	z2.Invert(&z2)
	x2.Multiply(&x2, &z2)
	copy(dst[:], x2.Bytes())
}

func scalarBaseMult(dst, scalar *[32]byte) {
	checkBasepoint()
	scalarMult(dst, scalar, &basePoint)
}

func x25519(dst *[32]byte, scalar, point []byte) ([]byte, error) {
	var in [32]byte
	if l := len(scalar); l != 32 {
		return nil, errors.New("bad scalar length: " + strconv.Itoa(l) + ", expected 32")
	}
	if l := len(point); l != 32 {
		return nil, errors.New("bad point length: " + strconv.Itoa(l) + ", expected 32")
	}
	copy(in[:], scalar)
	if &point[0] == &Basepoint[0] {
		scalarBaseMult(dst, &in)
	} else {
		var base, zero [32]byte
		copy(base[:], point)
		scalarMult(dst, &in, &base)
		if subtle.ConstantTimeCompare(dst[:], zero[:]) == 1 {
			return nil, errors.New("bad input point: low order point")
		}
	}
	return dst[:], nil
}

func checkBasepoint() {
	if subtle.ConstantTimeCompare(Basepoint, []byte{
		0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	}) != 1 {
		panic("curve25519: global Basepoint value was modified")
	}
}
