// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bn256

import (
	"math/big"
)

// curvePoint implements the elliptic curve y²=x³+3. Points are kept in
// Jacobian form and t=z² when valid. G₁ is the set of points of this curve on
// GF(p).
type curvePoint struct {
	x, y, z, t *big.Int
}

var curveB = new(big.Int).SetInt64(3)

// curveGen is the generator of G₁.
var curveGen = &curvePoint{
	new(big.Int).SetInt64(1),
	new(big.Int).SetInt64(-2),
	new(big.Int).SetInt64(1),
	new(big.Int).SetInt64(1),
}

func newCurvePoint(pool *bnPool) *curvePoint {
	return &curvePoint{
		pool.Get(),
		pool.Get(),
		pool.Get(),
		pool.Get(),
	}
}

func (c *curvePoint) String() string {
	c.MakeAffine(new(bnPool))
	return "(" + c.x.String() + ", " + c.y.String() + ")"
}

func (c *curvePoint) Put(pool *bnPool) {
	pool.Put(c.x)
	pool.Put(c.y)
	pool.Put(c.z)
	pool.Put(c.t)
}

func (c *curvePoint) Set(a *curvePoint) {
	c.x.Set(a.x)
	c.y.Set(a.y)
	c.z.Set(a.z)
	c.t.Set(a.t)
}

// IsOnCurve returns true iff c is on the curve where c must be in affine form.
func (c *curvePoint) IsOnCurve() bool {
	yy := new(big.Int).Mul(c.y, c.y)
	xxx := new(big.Int).Mul(c.x, c.x)
	xxx.Mul(xxx, c.x)
	yy.Sub(yy, xxx)
	yy.Sub(yy, curveB)
	if yy.Sign() < 0 || yy.Cmp(p) >= 0 {
		yy.Mod(yy, p)
	}
	return yy.Sign() == 0
}

func (c *curvePoint) SetInfinity() {
	c.z.SetInt64(0)
}

func (c *curvePoint) IsInfinity() bool {
	return c.z.Sign() == 0
}

func (c *curvePoint) Add(a, b *curvePoint, pool *bnPool) {
	if a.IsInfinity() {
		c.Set(b)
		return
	}
	if b.IsInfinity() {
		c.Set(a)
		return
	}

	// See http://hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-0/addition/add-2007-bl.op3

	// Normalize the points by replacing a = [x1:y1:z1] and b = [x2:y2:z2]
	// by [u1:s1:z1·z2] and [u2:s2:z1·z2]
	// where u1 = x1·z2², s1 = y1·z2³ and u1 = x2·z1², s2 = y2·z1³
	z1z1 := pool.Get().Mul(a.z, a.z)
	z1z1.Mod(z1z1, p)
	z2z2 := pool.Get().Mul(b.z, b.z)
	z2z2.Mod(z2z2, p)
	u1 := pool.Get().Mul(a.x, z2z2)
	u1.Mod(u1, p)
	u2 := pool.Get().Mul(b.x, z1z1)
	u2.Mod(u2, p)

	t := pool.Get().Mul(b.z, z2z2)
	t.Mod(t, p)
	s1 := pool.Get().Mul(a.y, t)
	s1.Mod(s1, p)

	t.Mul(a.z, z1z1)
	t.Mod(t, p)
	s2 := pool.Get().Mul(b.y, t)
	s2.Mod(s2, p)

	// Compute x = (2h)²(s²-u1-u2)
	// where s = (s2-s1)/(u2-u1) is the slope of the line through
	// (u1,s1) and (u2,s2). The extra factor 2h = 2(u2-u1) comes from the value of z below.
	// This is also:
	// 4(s2-s1)² - 4h²(u1+u2) = 4(s2-s1)² - 4h³ - 4h²(2u1)
	//                        = r² - j - 2v
	// with the notations below.
	h := pool.Get().Sub(u2, u1)
	xEqual := h.Sign() == 0

	t.Add(h, h)
	// i = 4h²
	i := pool.Get().Mul(t, t)
	i.Mod(i, p)
	// j = 4h³
	j := pool.Get().Mul(h, i)
	j.Mod(j, p)

	t.Sub(s2, s1)
	yEqual := t.Sign() == 0
	if xEqual && yEqual {
		c.Double(a, pool)
		return
	}
	r := pool.Get().Add(t, t)

	v := pool.Get().Mul(u1, i)
	v.Mod(v, p)

	// t4 = 4(s2-s1)²
	t4 := pool.Get().Mul(r, r)
	t4.Mod(t4, p)
	t.Add(v, v)
	t6 := pool.Get().Sub(t4, j)
	c.x.Sub(t6, t)

	// Set y = -(2h)³(s1 + s*(x/4h²-u1))
	// This is also
	// y = - 2·s1·j - (s2-s1)(2x - 2i·u1) = r(v-x) - 2·s1·j
	t.Sub(v, c.x) // t7
	t4.Mul(s1, j) // t8
	t4.Mod(t4, p)
	t6.Add(t4, t4) // t9
	t4.Mul(r, t)   // t10
	t4.Mod(t4, p)
	c.y.Sub(t4, t6)

	// Set z = 2(u2-u1)·z1·z2 = 2h·z1·z2
	t.Add(a.z, b.z) // t11
	t4.Mul(t, t)    // t12
	t4.Mod(t4, p)
	t.Sub(t4, z1z1) // t13
	t4.Sub(t, z2z2) // t14
	c.z.Mul(t4, h)
	c.z.Mod(c.z, p)

	pool.Put(z1z1)
	pool.Put(z2z2)
	pool.Put(u1)
	pool.Put(u2)
	pool.Put(t)
	pool.Put(s1)
	pool.Put(s2)
	pool.Put(h)
	pool.Put(i)
	pool.Put(j)
	pool.Put(r)
	pool.Put(v)
	pool.Put(t4)
	pool.Put(t6)
}

func (c *curvePoint) Double(a *curvePoint, pool *bnPool) {
	// See http://hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-0/doubling/dbl-2009-l.op3
	A := pool.Get().Mul(a.x, a.x)
	A.Mod(A, p)
	B := pool.Get().Mul(a.y, a.y)
	B.Mod(B, p)
	C := pool.Get().Mul(B, B)
	C.Mod(C, p)

	t := pool.Get().Add(a.x, B)
	t2 := pool.Get().Mul(t, t)
	t2.Mod(t2, p)
	t.Sub(t2, A)
	t2.Sub(t, C)
	d := pool.Get().Add(t2, t2)
	t.Add(A, A)
	e := pool.Get().Add(t, A)
	f := pool.Get().Mul(e, e)
	f.Mod(f, p)

	t.Add(d, d)
	c.x.Sub(f, t)

	t.Add(C, C)
	t2.Add(t, t)
	t.Add(t2, t2)
	c.y.Sub(d, c.x)
	t2.Mul(e, c.y)
	t2.Mod(t2, p)
	c.y.Sub(t2, t)

	t.Mul(a.y, a.z)
	t.Mod(t, p)
	c.z.Add(t, t)

	pool.Put(A)
	pool.Put(B)
	pool.Put(C)
	pool.Put(t)
	pool.Put(t2)
	pool.Put(d)
	pool.Put(e)
	pool.Put(f)
}

func (c *curvePoint) Mul(a *curvePoint, scalar *big.Int, pool *bnPool) *curvePoint {
	sum := newCurvePoint(pool)
	sum.SetInfinity()
	t := newCurvePoint(pool)

	for i := scalar.BitLen(); i >= 0; i-- {
		t.Double(sum, pool)
		if scalar.Bit(i) != 0 {
			sum.Add(t, a, pool)
		} else {
			sum.Set(t)
		}
	}

	c.Set(sum)
	sum.Put(pool)
	t.Put(pool)
	return c
}

// MakeAffine converts c to affine form and returns c. If c is ∞, then it sets
// c to 0 : 1 : 0.
func (c *curvePoint) MakeAffine(pool *bnPool) *curvePoint {
	if words := c.z.Bits(); len(words) == 1 && words[0] == 1 {
		return c
	}
	if c.IsInfinity() {
		c.x.SetInt64(0)
		c.y.SetInt64(1)
		c.z.SetInt64(0)
		c.t.SetInt64(0)
		return c
	}

	zInv := pool.Get().ModInverse(c.z, p)
	t := pool.Get().Mul(c.y, zInv)
	t.Mod(t, p)
	zInv2 := pool.Get().Mul(zInv, zInv)
	zInv2.Mod(zInv2, p)
	c.y.Mul(t, zInv2)
	c.y.Mod(c.y, p)
	t.Mul(c.x, zInv2)
	t.Mod(t, p)
	c.x.Set(t)
	c.z.SetInt64(1)
	c.t.SetInt64(1)

	pool.Put(zInv)
	pool.Put(t)
	pool.Put(zInv2)

	return c
}

func (c *curvePoint) Negative(a *curvePoint) {
	c.x.Set(a.x)
	c.y.Neg(a.y)
	c.z.Set(a.z)
	c.t.SetInt64(0)
}
