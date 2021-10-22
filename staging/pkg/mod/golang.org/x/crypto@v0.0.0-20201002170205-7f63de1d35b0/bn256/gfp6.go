// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bn256

// For details of the algorithms used, see "Multiplication and Squaring on
// Pairing-Friendly Fields, Devegili et al.
// http://eprint.iacr.org/2006/471.pdf.

import (
	"math/big"
)

// gfP6 implements the field of size p⁶ as a cubic extension of gfP2 where τ³=ξ
// and ξ=i+3.
type gfP6 struct {
	x, y, z *gfP2 // value is xτ² + yτ + z
}

func newGFp6(pool *bnPool) *gfP6 {
	return &gfP6{newGFp2(pool), newGFp2(pool), newGFp2(pool)}
}

func (e *gfP6) String() string {
	return "(" + e.x.String() + "," + e.y.String() + "," + e.z.String() + ")"
}

func (e *gfP6) Put(pool *bnPool) {
	e.x.Put(pool)
	e.y.Put(pool)
	e.z.Put(pool)
}

func (e *gfP6) Set(a *gfP6) *gfP6 {
	e.x.Set(a.x)
	e.y.Set(a.y)
	e.z.Set(a.z)
	return e
}

func (e *gfP6) SetZero() *gfP6 {
	e.x.SetZero()
	e.y.SetZero()
	e.z.SetZero()
	return e
}

func (e *gfP6) SetOne() *gfP6 {
	e.x.SetZero()
	e.y.SetZero()
	e.z.SetOne()
	return e
}

func (e *gfP6) Minimal() {
	e.x.Minimal()
	e.y.Minimal()
	e.z.Minimal()
}

func (e *gfP6) IsZero() bool {
	return e.x.IsZero() && e.y.IsZero() && e.z.IsZero()
}

func (e *gfP6) IsOne() bool {
	return e.x.IsZero() && e.y.IsZero() && e.z.IsOne()
}

func (e *gfP6) Negative(a *gfP6) *gfP6 {
	e.x.Negative(a.x)
	e.y.Negative(a.y)
	e.z.Negative(a.z)
	return e
}

func (e *gfP6) Frobenius(a *gfP6, pool *bnPool) *gfP6 {
	e.x.Conjugate(a.x)
	e.y.Conjugate(a.y)
	e.z.Conjugate(a.z)

	e.x.Mul(e.x, xiTo2PMinus2Over3, pool)
	e.y.Mul(e.y, xiToPMinus1Over3, pool)
	return e
}

// FrobeniusP2 computes (xτ²+yτ+z)^(p²) = xτ^(2p²) + yτ^(p²) + z
func (e *gfP6) FrobeniusP2(a *gfP6) *gfP6 {
	// τ^(2p²) = τ²τ^(2p²-2) = τ²ξ^((2p²-2)/3)
	e.x.MulScalar(a.x, xiTo2PSquaredMinus2Over3)
	// τ^(p²) = ττ^(p²-1) = τξ^((p²-1)/3)
	e.y.MulScalar(a.y, xiToPSquaredMinus1Over3)
	e.z.Set(a.z)
	return e
}

func (e *gfP6) Add(a, b *gfP6) *gfP6 {
	e.x.Add(a.x, b.x)
	e.y.Add(a.y, b.y)
	e.z.Add(a.z, b.z)
	return e
}

func (e *gfP6) Sub(a, b *gfP6) *gfP6 {
	e.x.Sub(a.x, b.x)
	e.y.Sub(a.y, b.y)
	e.z.Sub(a.z, b.z)
	return e
}

func (e *gfP6) Double(a *gfP6) *gfP6 {
	e.x.Double(a.x)
	e.y.Double(a.y)
	e.z.Double(a.z)
	return e
}

func (e *gfP6) Mul(a, b *gfP6, pool *bnPool) *gfP6 {
	// "Multiplication and Squaring on Pairing-Friendly Fields"
	// Section 4, Karatsuba method.
	// http://eprint.iacr.org/2006/471.pdf

	v0 := newGFp2(pool)
	v0.Mul(a.z, b.z, pool)
	v1 := newGFp2(pool)
	v1.Mul(a.y, b.y, pool)
	v2 := newGFp2(pool)
	v2.Mul(a.x, b.x, pool)

	t0 := newGFp2(pool)
	t0.Add(a.x, a.y)
	t1 := newGFp2(pool)
	t1.Add(b.x, b.y)
	tz := newGFp2(pool)
	tz.Mul(t0, t1, pool)

	tz.Sub(tz, v1)
	tz.Sub(tz, v2)
	tz.MulXi(tz, pool)
	tz.Add(tz, v0)

	t0.Add(a.y, a.z)
	t1.Add(b.y, b.z)
	ty := newGFp2(pool)
	ty.Mul(t0, t1, pool)
	ty.Sub(ty, v0)
	ty.Sub(ty, v1)
	t0.MulXi(v2, pool)
	ty.Add(ty, t0)

	t0.Add(a.x, a.z)
	t1.Add(b.x, b.z)
	tx := newGFp2(pool)
	tx.Mul(t0, t1, pool)
	tx.Sub(tx, v0)
	tx.Add(tx, v1)
	tx.Sub(tx, v2)

	e.x.Set(tx)
	e.y.Set(ty)
	e.z.Set(tz)

	t0.Put(pool)
	t1.Put(pool)
	tx.Put(pool)
	ty.Put(pool)
	tz.Put(pool)
	v0.Put(pool)
	v1.Put(pool)
	v2.Put(pool)
	return e
}

func (e *gfP6) MulScalar(a *gfP6, b *gfP2, pool *bnPool) *gfP6 {
	e.x.Mul(a.x, b, pool)
	e.y.Mul(a.y, b, pool)
	e.z.Mul(a.z, b, pool)
	return e
}

func (e *gfP6) MulGFP(a *gfP6, b *big.Int) *gfP6 {
	e.x.MulScalar(a.x, b)
	e.y.MulScalar(a.y, b)
	e.z.MulScalar(a.z, b)
	return e
}

// MulTau computes τ·(aτ²+bτ+c) = bτ²+cτ+aξ
func (e *gfP6) MulTau(a *gfP6, pool *bnPool) {
	tz := newGFp2(pool)
	tz.MulXi(a.x, pool)
	ty := newGFp2(pool)
	ty.Set(a.y)
	e.y.Set(a.z)
	e.x.Set(ty)
	e.z.Set(tz)
	tz.Put(pool)
	ty.Put(pool)
}

func (e *gfP6) Square(a *gfP6, pool *bnPool) *gfP6 {
	v0 := newGFp2(pool).Square(a.z, pool)
	v1 := newGFp2(pool).Square(a.y, pool)
	v2 := newGFp2(pool).Square(a.x, pool)

	c0 := newGFp2(pool).Add(a.x, a.y)
	c0.Square(c0, pool)
	c0.Sub(c0, v1)
	c0.Sub(c0, v2)
	c0.MulXi(c0, pool)
	c0.Add(c0, v0)

	c1 := newGFp2(pool).Add(a.y, a.z)
	c1.Square(c1, pool)
	c1.Sub(c1, v0)
	c1.Sub(c1, v1)
	xiV2 := newGFp2(pool).MulXi(v2, pool)
	c1.Add(c1, xiV2)

	c2 := newGFp2(pool).Add(a.x, a.z)
	c2.Square(c2, pool)
	c2.Sub(c2, v0)
	c2.Add(c2, v1)
	c2.Sub(c2, v2)

	e.x.Set(c2)
	e.y.Set(c1)
	e.z.Set(c0)

	v0.Put(pool)
	v1.Put(pool)
	v2.Put(pool)
	c0.Put(pool)
	c1.Put(pool)
	c2.Put(pool)
	xiV2.Put(pool)

	return e
}

func (e *gfP6) Invert(a *gfP6, pool *bnPool) *gfP6 {
	// See "Implementing cryptographic pairings", M. Scott, section 3.2.
	// ftp://136.206.11.249/pub/crypto/pairings.pdf

	// Here we can give a short explanation of how it works: let j be a cubic root of
	// unity in GF(p²) so that 1+j+j²=0.
	// Then (xτ² + yτ + z)(xj²τ² + yjτ + z)(xjτ² + yj²τ + z)
	// = (xτ² + yτ + z)(Cτ²+Bτ+A)
	// = (x³ξ²+y³ξ+z³-3ξxyz) = F is an element of the base field (the norm).
	//
	// On the other hand (xj²τ² + yjτ + z)(xjτ² + yj²τ + z)
	// = τ²(y²-ξxz) + τ(ξx²-yz) + (z²-ξxy)
	//
	// So that's why A = (z²-ξxy), B = (ξx²-yz), C = (y²-ξxz)
	t1 := newGFp2(pool)

	A := newGFp2(pool)
	A.Square(a.z, pool)
	t1.Mul(a.x, a.y, pool)
	t1.MulXi(t1, pool)
	A.Sub(A, t1)

	B := newGFp2(pool)
	B.Square(a.x, pool)
	B.MulXi(B, pool)
	t1.Mul(a.y, a.z, pool)
	B.Sub(B, t1)

	C := newGFp2(pool)
	C.Square(a.y, pool)
	t1.Mul(a.x, a.z, pool)
	C.Sub(C, t1)

	F := newGFp2(pool)
	F.Mul(C, a.y, pool)
	F.MulXi(F, pool)
	t1.Mul(A, a.z, pool)
	F.Add(F, t1)
	t1.Mul(B, a.x, pool)
	t1.MulXi(t1, pool)
	F.Add(F, t1)

	F.Invert(F, pool)

	e.x.Mul(C, F, pool)
	e.y.Mul(B, F, pool)
	e.z.Mul(A, F, pool)

	t1.Put(pool)
	A.Put(pool)
	B.Put(pool)
	C.Put(pool)
	F.Put(pool)

	return e
}
