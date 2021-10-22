// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bn256

import (
	"math/big"
)

// twistPoint implements the elliptic curve y²=x³+3/ξ over GF(p²). Points are
// kept in Jacobian form and t=z² when valid. The group G₂ is the set of
// n-torsion points of this curve over GF(p²) (where n = Order)
type twistPoint struct {
	x, y, z, t *gfP2
}

var twistB = &gfP2{
	bigFromBase10("6500054969564660373279643874235990574282535810762300357187714502686418407178"),
	bigFromBase10("45500384786952622612957507119651934019977750675336102500314001518804928850249"),
}

// twistGen is the generator of group G₂.
var twistGen = &twistPoint{
	&gfP2{
		bigFromBase10("21167961636542580255011770066570541300993051739349375019639421053990175267184"),
		bigFromBase10("64746500191241794695844075326670126197795977525365406531717464316923369116492"),
	},
	&gfP2{
		bigFromBase10("20666913350058776956210519119118544732556678129809273996262322366050359951122"),
		bigFromBase10("17778617556404439934652658462602675281523610326338642107814333856843981424549"),
	},
	&gfP2{
		bigFromBase10("0"),
		bigFromBase10("1"),
	},
	&gfP2{
		bigFromBase10("0"),
		bigFromBase10("1"),
	},
}

func newTwistPoint(pool *bnPool) *twistPoint {
	return &twistPoint{
		newGFp2(pool),
		newGFp2(pool),
		newGFp2(pool),
		newGFp2(pool),
	}
}

func (c *twistPoint) String() string {
	return "(" + c.x.String() + ", " + c.y.String() + ", " + c.z.String() + ")"
}

func (c *twistPoint) Put(pool *bnPool) {
	c.x.Put(pool)
	c.y.Put(pool)
	c.z.Put(pool)
	c.t.Put(pool)
}

func (c *twistPoint) Set(a *twistPoint) {
	c.x.Set(a.x)
	c.y.Set(a.y)
	c.z.Set(a.z)
	c.t.Set(a.t)
}

// IsOnCurve returns true iff c is on the curve where c must be in affine form.
func (c *twistPoint) IsOnCurve() bool {
	pool := new(bnPool)
	yy := newGFp2(pool).Square(c.y, pool)
	xxx := newGFp2(pool).Square(c.x, pool)
	xxx.Mul(xxx, c.x, pool)
	yy.Sub(yy, xxx)
	yy.Sub(yy, twistB)
	yy.Minimal()
	return yy.x.Sign() == 0 && yy.y.Sign() == 0
}

func (c *twistPoint) SetInfinity() {
	c.z.SetZero()
}

func (c *twistPoint) IsInfinity() bool {
	return c.z.IsZero()
}

func (c *twistPoint) Add(a, b *twistPoint, pool *bnPool) {
	// For additional comments, see the same function in curve.go.

	if a.IsInfinity() {
		c.Set(b)
		return
	}
	if b.IsInfinity() {
		c.Set(a)
		return
	}

	// See http://hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-0/addition/add-2007-bl.op3
	z1z1 := newGFp2(pool).Square(a.z, pool)
	z2z2 := newGFp2(pool).Square(b.z, pool)
	u1 := newGFp2(pool).Mul(a.x, z2z2, pool)
	u2 := newGFp2(pool).Mul(b.x, z1z1, pool)

	t := newGFp2(pool).Mul(b.z, z2z2, pool)
	s1 := newGFp2(pool).Mul(a.y, t, pool)

	t.Mul(a.z, z1z1, pool)
	s2 := newGFp2(pool).Mul(b.y, t, pool)

	h := newGFp2(pool).Sub(u2, u1)
	xEqual := h.IsZero()

	t.Add(h, h)
	i := newGFp2(pool).Square(t, pool)
	j := newGFp2(pool).Mul(h, i, pool)

	t.Sub(s2, s1)
	yEqual := t.IsZero()
	if xEqual && yEqual {
		c.Double(a, pool)
		return
	}
	r := newGFp2(pool).Add(t, t)

	v := newGFp2(pool).Mul(u1, i, pool)

	t4 := newGFp2(pool).Square(r, pool)
	t.Add(v, v)
	t6 := newGFp2(pool).Sub(t4, j)
	c.x.Sub(t6, t)

	t.Sub(v, c.x)       // t7
	t4.Mul(s1, j, pool) // t8
	t6.Add(t4, t4)      // t9
	t4.Mul(r, t, pool)  // t10
	c.y.Sub(t4, t6)

	t.Add(a.z, b.z)    // t11
	t4.Square(t, pool) // t12
	t.Sub(t4, z1z1)    // t13
	t4.Sub(t, z2z2)    // t14
	c.z.Mul(t4, h, pool)

	z1z1.Put(pool)
	z2z2.Put(pool)
	u1.Put(pool)
	u2.Put(pool)
	t.Put(pool)
	s1.Put(pool)
	s2.Put(pool)
	h.Put(pool)
	i.Put(pool)
	j.Put(pool)
	r.Put(pool)
	v.Put(pool)
	t4.Put(pool)
	t6.Put(pool)
}

func (c *twistPoint) Double(a *twistPoint, pool *bnPool) {
	// See http://hyperelliptic.org/EFD/g1p/auto-code/shortw/jacobian-0/doubling/dbl-2009-l.op3
	A := newGFp2(pool).Square(a.x, pool)
	B := newGFp2(pool).Square(a.y, pool)
	C := newGFp2(pool).Square(B, pool)

	t := newGFp2(pool).Add(a.x, B)
	t2 := newGFp2(pool).Square(t, pool)
	t.Sub(t2, A)
	t2.Sub(t, C)
	d := newGFp2(pool).Add(t2, t2)
	t.Add(A, A)
	e := newGFp2(pool).Add(t, A)
	f := newGFp2(pool).Square(e, pool)

	t.Add(d, d)
	c.x.Sub(f, t)

	t.Add(C, C)
	t2.Add(t, t)
	t.Add(t2, t2)
	c.y.Sub(d, c.x)
	t2.Mul(e, c.y, pool)
	c.y.Sub(t2, t)

	t.Mul(a.y, a.z, pool)
	c.z.Add(t, t)

	A.Put(pool)
	B.Put(pool)
	C.Put(pool)
	t.Put(pool)
	t2.Put(pool)
	d.Put(pool)
	e.Put(pool)
	f.Put(pool)
}

func (c *twistPoint) Mul(a *twistPoint, scalar *big.Int, pool *bnPool) *twistPoint {
	sum := newTwistPoint(pool)
	sum.SetInfinity()
	t := newTwistPoint(pool)

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
func (c *twistPoint) MakeAffine(pool *bnPool) *twistPoint {
	if c.z.IsOne() {
		return c
	}
	if c.IsInfinity() {
		c.x.SetZero()
		c.y.SetOne()
		c.z.SetZero()
		c.t.SetZero()
		return c
	}

	zInv := newGFp2(pool).Invert(c.z, pool)
	t := newGFp2(pool).Mul(c.y, zInv, pool)
	zInv2 := newGFp2(pool).Square(zInv, pool)
	c.y.Mul(t, zInv2, pool)
	t.Mul(c.x, zInv2, pool)
	c.x.Set(t)
	c.z.SetOne()
	c.t.SetOne()

	zInv.Put(pool)
	t.Put(pool)
	zInv2.Put(pool)

	return c
}

func (c *twistPoint) Negative(a *twistPoint, pool *bnPool) {
	c.x.Set(a.x)
	c.y.SetZero()
	c.y.Sub(c.y, a.y)
	c.z.Set(a.z)
	c.t.SetZero()
}
