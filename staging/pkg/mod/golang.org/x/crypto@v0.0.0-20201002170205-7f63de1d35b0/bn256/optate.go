// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bn256

func lineFunctionAdd(r, p *twistPoint, q *curvePoint, r2 *gfP2, pool *bnPool) (a, b, c *gfP2, rOut *twistPoint) {
	// See the mixed addition algorithm from "Faster Computation of the
	// Tate Pairing", http://arxiv.org/pdf/0904.0854v3.pdf

	B := newGFp2(pool).Mul(p.x, r.t, pool)

	D := newGFp2(pool).Add(p.y, r.z)
	D.Square(D, pool)
	D.Sub(D, r2)
	D.Sub(D, r.t)
	D.Mul(D, r.t, pool)

	H := newGFp2(pool).Sub(B, r.x)
	I := newGFp2(pool).Square(H, pool)

	E := newGFp2(pool).Add(I, I)
	E.Add(E, E)

	J := newGFp2(pool).Mul(H, E, pool)

	L1 := newGFp2(pool).Sub(D, r.y)
	L1.Sub(L1, r.y)

	V := newGFp2(pool).Mul(r.x, E, pool)

	rOut = newTwistPoint(pool)
	rOut.x.Square(L1, pool)
	rOut.x.Sub(rOut.x, J)
	rOut.x.Sub(rOut.x, V)
	rOut.x.Sub(rOut.x, V)

	rOut.z.Add(r.z, H)
	rOut.z.Square(rOut.z, pool)
	rOut.z.Sub(rOut.z, r.t)
	rOut.z.Sub(rOut.z, I)

	t := newGFp2(pool).Sub(V, rOut.x)
	t.Mul(t, L1, pool)
	t2 := newGFp2(pool).Mul(r.y, J, pool)
	t2.Add(t2, t2)
	rOut.y.Sub(t, t2)

	rOut.t.Square(rOut.z, pool)

	t.Add(p.y, rOut.z)
	t.Square(t, pool)
	t.Sub(t, r2)
	t.Sub(t, rOut.t)

	t2.Mul(L1, p.x, pool)
	t2.Add(t2, t2)
	a = newGFp2(pool)
	a.Sub(t2, t)

	c = newGFp2(pool)
	c.MulScalar(rOut.z, q.y)
	c.Add(c, c)

	b = newGFp2(pool)
	b.SetZero()
	b.Sub(b, L1)
	b.MulScalar(b, q.x)
	b.Add(b, b)

	B.Put(pool)
	D.Put(pool)
	H.Put(pool)
	I.Put(pool)
	E.Put(pool)
	J.Put(pool)
	L1.Put(pool)
	V.Put(pool)
	t.Put(pool)
	t2.Put(pool)

	return
}

func lineFunctionDouble(r *twistPoint, q *curvePoint, pool *bnPool) (a, b, c *gfP2, rOut *twistPoint) {
	// See the doubling algorithm for a=0 from "Faster Computation of the
	// Tate Pairing", http://arxiv.org/pdf/0904.0854v3.pdf

	A := newGFp2(pool).Square(r.x, pool)
	B := newGFp2(pool).Square(r.y, pool)
	C := newGFp2(pool).Square(B, pool)

	D := newGFp2(pool).Add(r.x, B)
	D.Square(D, pool)
	D.Sub(D, A)
	D.Sub(D, C)
	D.Add(D, D)

	E := newGFp2(pool).Add(A, A)
	E.Add(E, A)

	G := newGFp2(pool).Square(E, pool)

	rOut = newTwistPoint(pool)
	rOut.x.Sub(G, D)
	rOut.x.Sub(rOut.x, D)

	rOut.z.Add(r.y, r.z)
	rOut.z.Square(rOut.z, pool)
	rOut.z.Sub(rOut.z, B)
	rOut.z.Sub(rOut.z, r.t)

	rOut.y.Sub(D, rOut.x)
	rOut.y.Mul(rOut.y, E, pool)
	t := newGFp2(pool).Add(C, C)
	t.Add(t, t)
	t.Add(t, t)
	rOut.y.Sub(rOut.y, t)

	rOut.t.Square(rOut.z, pool)

	t.Mul(E, r.t, pool)
	t.Add(t, t)
	b = newGFp2(pool)
	b.SetZero()
	b.Sub(b, t)
	b.MulScalar(b, q.x)

	a = newGFp2(pool)
	a.Add(r.x, E)
	a.Square(a, pool)
	a.Sub(a, A)
	a.Sub(a, G)
	t.Add(B, B)
	t.Add(t, t)
	a.Sub(a, t)

	c = newGFp2(pool)
	c.Mul(rOut.z, r.t, pool)
	c.Add(c, c)
	c.MulScalar(c, q.y)

	A.Put(pool)
	B.Put(pool)
	C.Put(pool)
	D.Put(pool)
	E.Put(pool)
	G.Put(pool)
	t.Put(pool)

	return
}

func mulLine(ret *gfP12, a, b, c *gfP2, pool *bnPool) {
	a2 := newGFp6(pool)
	a2.x.SetZero()
	a2.y.Set(a)
	a2.z.Set(b)
	a2.Mul(a2, ret.x, pool)
	t3 := newGFp6(pool).MulScalar(ret.y, c, pool)

	t := newGFp2(pool)
	t.Add(b, c)
	t2 := newGFp6(pool)
	t2.x.SetZero()
	t2.y.Set(a)
	t2.z.Set(t)
	ret.x.Add(ret.x, ret.y)

	ret.y.Set(t3)

	ret.x.Mul(ret.x, t2, pool)
	ret.x.Sub(ret.x, a2)
	ret.x.Sub(ret.x, ret.y)
	a2.MulTau(a2, pool)
	ret.y.Add(ret.y, a2)

	a2.Put(pool)
	t3.Put(pool)
	t2.Put(pool)
	t.Put(pool)
}

// sixuPlus2NAF is 6u+2 in non-adjacent form.
var sixuPlus2NAF = []int8{0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 1}

// miller implements the Miller loop for calculating the Optimal Ate pairing.
// See algorithm 1 from http://cryptojedi.org/papers/dclxvi-20100714.pdf
func miller(q *twistPoint, p *curvePoint, pool *bnPool) *gfP12 {
	ret := newGFp12(pool)
	ret.SetOne()

	aAffine := newTwistPoint(pool)
	aAffine.Set(q)
	aAffine.MakeAffine(pool)

	bAffine := newCurvePoint(pool)
	bAffine.Set(p)
	bAffine.MakeAffine(pool)

	minusA := newTwistPoint(pool)
	minusA.Negative(aAffine, pool)

	r := newTwistPoint(pool)
	r.Set(aAffine)

	r2 := newGFp2(pool)
	r2.Square(aAffine.y, pool)

	for i := len(sixuPlus2NAF) - 1; i > 0; i-- {
		a, b, c, newR := lineFunctionDouble(r, bAffine, pool)
		if i != len(sixuPlus2NAF)-1 {
			ret.Square(ret, pool)
		}

		mulLine(ret, a, b, c, pool)
		a.Put(pool)
		b.Put(pool)
		c.Put(pool)
		r.Put(pool)
		r = newR

		switch sixuPlus2NAF[i-1] {
		case 1:
			a, b, c, newR = lineFunctionAdd(r, aAffine, bAffine, r2, pool)
		case -1:
			a, b, c, newR = lineFunctionAdd(r, minusA, bAffine, r2, pool)
		default:
			continue
		}

		mulLine(ret, a, b, c, pool)
		a.Put(pool)
		b.Put(pool)
		c.Put(pool)
		r.Put(pool)
		r = newR
	}

	// In order to calculate Q1 we have to convert q from the sextic twist
	// to the full GF(p^12) group, apply the Frobenius there, and convert
	// back.
	//
	// The twist isomorphism is (x', y') -> (xω², yω³). If we consider just
	// x for a moment, then after applying the Frobenius, we have x̄ω^(2p)
	// where x̄ is the conjugate of x. If we are going to apply the inverse
	// isomorphism we need a value with a single coefficient of ω² so we
	// rewrite this as x̄ω^(2p-2)ω². ξ⁶ = ω and, due to the construction of
	// p, 2p-2 is a multiple of six. Therefore we can rewrite as
	// x̄ξ^((p-1)/3)ω² and applying the inverse isomorphism eliminates the
	// ω².
	//
	// A similar argument can be made for the y value.

	q1 := newTwistPoint(pool)
	q1.x.Conjugate(aAffine.x)
	q1.x.Mul(q1.x, xiToPMinus1Over3, pool)
	q1.y.Conjugate(aAffine.y)
	q1.y.Mul(q1.y, xiToPMinus1Over2, pool)
	q1.z.SetOne()
	q1.t.SetOne()

	// For Q2 we are applying the p² Frobenius. The two conjugations cancel
	// out and we are left only with the factors from the isomorphism. In
	// the case of x, we end up with a pure number which is why
	// xiToPSquaredMinus1Over3 is ∈ GF(p). With y we get a factor of -1. We
	// ignore this to end up with -Q2.

	minusQ2 := newTwistPoint(pool)
	minusQ2.x.MulScalar(aAffine.x, xiToPSquaredMinus1Over3)
	minusQ2.y.Set(aAffine.y)
	minusQ2.z.SetOne()
	minusQ2.t.SetOne()

	r2.Square(q1.y, pool)
	a, b, c, newR := lineFunctionAdd(r, q1, bAffine, r2, pool)
	mulLine(ret, a, b, c, pool)
	a.Put(pool)
	b.Put(pool)
	c.Put(pool)
	r.Put(pool)
	r = newR

	r2.Square(minusQ2.y, pool)
	a, b, c, newR = lineFunctionAdd(r, minusQ2, bAffine, r2, pool)
	mulLine(ret, a, b, c, pool)
	a.Put(pool)
	b.Put(pool)
	c.Put(pool)
	r.Put(pool)
	r = newR

	aAffine.Put(pool)
	bAffine.Put(pool)
	minusA.Put(pool)
	r.Put(pool)
	r2.Put(pool)

	return ret
}

// finalExponentiation computes the (p¹²-1)/Order-th power of an element of
// GF(p¹²) to obtain an element of GT (steps 13-15 of algorithm 1 from
// http://cryptojedi.org/papers/dclxvi-20100714.pdf)
func finalExponentiation(in *gfP12, pool *bnPool) *gfP12 {
	t1 := newGFp12(pool)

	// This is the p^6-Frobenius
	t1.x.Negative(in.x)
	t1.y.Set(in.y)

	inv := newGFp12(pool)
	inv.Invert(in, pool)
	t1.Mul(t1, inv, pool)

	t2 := newGFp12(pool).FrobeniusP2(t1, pool)
	t1.Mul(t1, t2, pool)

	fp := newGFp12(pool).Frobenius(t1, pool)
	fp2 := newGFp12(pool).FrobeniusP2(t1, pool)
	fp3 := newGFp12(pool).Frobenius(fp2, pool)

	fu, fu2, fu3 := newGFp12(pool), newGFp12(pool), newGFp12(pool)
	fu.Exp(t1, u, pool)
	fu2.Exp(fu, u, pool)
	fu3.Exp(fu2, u, pool)

	y3 := newGFp12(pool).Frobenius(fu, pool)
	fu2p := newGFp12(pool).Frobenius(fu2, pool)
	fu3p := newGFp12(pool).Frobenius(fu3, pool)
	y2 := newGFp12(pool).FrobeniusP2(fu2, pool)

	y0 := newGFp12(pool)
	y0.Mul(fp, fp2, pool)
	y0.Mul(y0, fp3, pool)

	y1, y4, y5 := newGFp12(pool), newGFp12(pool), newGFp12(pool)
	y1.Conjugate(t1)
	y5.Conjugate(fu2)
	y3.Conjugate(y3)
	y4.Mul(fu, fu2p, pool)
	y4.Conjugate(y4)

	y6 := newGFp12(pool)
	y6.Mul(fu3, fu3p, pool)
	y6.Conjugate(y6)

	t0 := newGFp12(pool)
	t0.Square(y6, pool)
	t0.Mul(t0, y4, pool)
	t0.Mul(t0, y5, pool)
	t1.Mul(y3, y5, pool)
	t1.Mul(t1, t0, pool)
	t0.Mul(t0, y2, pool)
	t1.Square(t1, pool)
	t1.Mul(t1, t0, pool)
	t1.Square(t1, pool)
	t0.Mul(t1, y1, pool)
	t1.Mul(t1, y0, pool)
	t0.Square(t0, pool)
	t0.Mul(t0, t1, pool)

	inv.Put(pool)
	t1.Put(pool)
	t2.Put(pool)
	fp.Put(pool)
	fp2.Put(pool)
	fp3.Put(pool)
	fu.Put(pool)
	fu2.Put(pool)
	fu3.Put(pool)
	fu2p.Put(pool)
	fu3p.Put(pool)
	y0.Put(pool)
	y1.Put(pool)
	y2.Put(pool)
	y3.Put(pool)
	y4.Put(pool)
	y5.Put(pool)
	y6.Put(pool)

	return t0
}

func optimalAte(a *twistPoint, b *curvePoint, pool *bnPool) *gfP12 {
	e := miller(a, b, pool)
	ret := finalExponentiation(e, pool)
	e.Put(pool)

	if a.IsInfinity() || b.IsInfinity() {
		ret.SetOne()
	}

	return ret
}
