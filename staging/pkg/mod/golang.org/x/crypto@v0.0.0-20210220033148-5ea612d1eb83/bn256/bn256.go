// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bn256 implements a particular bilinear group.
//
// Bilinear groups are the basis of many of the new cryptographic protocols
// that have been proposed over the past decade. They consist of a triplet of
// groups (G₁, G₂ and GT) such that there exists a function e(g₁ˣ,g₂ʸ)=gTˣʸ
// (where gₓ is a generator of the respective group). That function is called
// a pairing function.
//
// This package specifically implements the Optimal Ate pairing over a 256-bit
// Barreto-Naehrig curve as described in
// http://cryptojedi.org/papers/dclxvi-20100714.pdf. Its output is compatible
// with the implementation described in that paper.
//
// This package previously claimed to operate at a 128-bit security level.
// However, recent improvements in attacks mean that is no longer true. See
// https://moderncrypto.org/mail-archive/curves/2016/000740.html.
//
// Deprecated: due to its weakened security, new systems should not rely on this
// elliptic curve. This package is frozen, and not implemented in constant time.
// There is a more complete implementation at github.com/cloudflare/bn256, but
// note that it suffers from the same security issues of the underlying curve.
package bn256 // import "golang.org/x/crypto/bn256"

import (
	"crypto/rand"
	"io"
	"math/big"
)

// G1 is an abstract cyclic group. The zero value is suitable for use as the
// output of an operation, but cannot be used as an input.
type G1 struct {
	p *curvePoint
}

// RandomG1 returns x and g₁ˣ where x is a random, non-zero number read from r.
func RandomG1(r io.Reader) (*big.Int, *G1, error) {
	var k *big.Int
	var err error

	for {
		k, err = rand.Int(r, Order)
		if err != nil {
			return nil, nil, err
		}
		if k.Sign() > 0 {
			break
		}
	}

	return k, new(G1).ScalarBaseMult(k), nil
}

func (e *G1) String() string {
	if e.p == nil {
		return "bn256.G1" + newCurvePoint(nil).String()
	}
	return "bn256.G1" + e.p.String()
}

// ScalarBaseMult sets e to g*k where g is the generator of the group and
// then returns e.
func (e *G1) ScalarBaseMult(k *big.Int) *G1 {
	if e.p == nil {
		e.p = newCurvePoint(nil)
	}
	e.p.Mul(curveGen, k, new(bnPool))
	return e
}

// ScalarMult sets e to a*k and then returns e.
func (e *G1) ScalarMult(a *G1, k *big.Int) *G1 {
	if e.p == nil {
		e.p = newCurvePoint(nil)
	}
	e.p.Mul(a.p, k, new(bnPool))
	return e
}

// Add sets e to a+b and then returns e.
//
// Warning: this function is not complete, it fails for a equal to b.
func (e *G1) Add(a, b *G1) *G1 {
	if e.p == nil {
		e.p = newCurvePoint(nil)
	}
	e.p.Add(a.p, b.p, new(bnPool))
	return e
}

// Neg sets e to -a and then returns e.
func (e *G1) Neg(a *G1) *G1 {
	if e.p == nil {
		e.p = newCurvePoint(nil)
	}
	e.p.Negative(a.p)
	return e
}

// Marshal converts n to a byte slice.
func (e *G1) Marshal() []byte {
	// Each value is a 256-bit number.
	const numBytes = 256 / 8

	if e.p.IsInfinity() {
		return make([]byte, numBytes*2)
	}

	e.p.MakeAffine(nil)

	xBytes := new(big.Int).Mod(e.p.x, p).Bytes()
	yBytes := new(big.Int).Mod(e.p.y, p).Bytes()

	ret := make([]byte, numBytes*2)
	copy(ret[1*numBytes-len(xBytes):], xBytes)
	copy(ret[2*numBytes-len(yBytes):], yBytes)

	return ret
}

// Unmarshal sets e to the result of converting the output of Marshal back into
// a group element and then returns e.
func (e *G1) Unmarshal(m []byte) (*G1, bool) {
	// Each value is a 256-bit number.
	const numBytes = 256 / 8

	if len(m) != 2*numBytes {
		return nil, false
	}

	if e.p == nil {
		e.p = newCurvePoint(nil)
	}

	e.p.x.SetBytes(m[0*numBytes : 1*numBytes])
	e.p.y.SetBytes(m[1*numBytes : 2*numBytes])

	if e.p.x.Sign() == 0 && e.p.y.Sign() == 0 {
		// This is the point at infinity.
		e.p.y.SetInt64(1)
		e.p.z.SetInt64(0)
		e.p.t.SetInt64(0)
	} else {
		e.p.z.SetInt64(1)
		e.p.t.SetInt64(1)

		if !e.p.IsOnCurve() {
			return nil, false
		}
	}

	return e, true
}

// G2 is an abstract cyclic group. The zero value is suitable for use as the
// output of an operation, but cannot be used as an input.
type G2 struct {
	p *twistPoint
}

// RandomG1 returns x and g₂ˣ where x is a random, non-zero number read from r.
func RandomG2(r io.Reader) (*big.Int, *G2, error) {
	var k *big.Int
	var err error

	for {
		k, err = rand.Int(r, Order)
		if err != nil {
			return nil, nil, err
		}
		if k.Sign() > 0 {
			break
		}
	}

	return k, new(G2).ScalarBaseMult(k), nil
}

func (e *G2) String() string {
	if e.p == nil {
		return "bn256.G2" + newTwistPoint(nil).String()
	}
	return "bn256.G2" + e.p.String()
}

// ScalarBaseMult sets e to g*k where g is the generator of the group and
// then returns out.
func (e *G2) ScalarBaseMult(k *big.Int) *G2 {
	if e.p == nil {
		e.p = newTwistPoint(nil)
	}
	e.p.Mul(twistGen, k, new(bnPool))
	return e
}

// ScalarMult sets e to a*k and then returns e.
func (e *G2) ScalarMult(a *G2, k *big.Int) *G2 {
	if e.p == nil {
		e.p = newTwistPoint(nil)
	}
	e.p.Mul(a.p, k, new(bnPool))
	return e
}

// Add sets e to a+b and then returns e.
//
// Warning: this function is not complete, it fails for a equal to b.
func (e *G2) Add(a, b *G2) *G2 {
	if e.p == nil {
		e.p = newTwistPoint(nil)
	}
	e.p.Add(a.p, b.p, new(bnPool))
	return e
}

// Marshal converts n into a byte slice.
func (n *G2) Marshal() []byte {
	// Each value is a 256-bit number.
	const numBytes = 256 / 8

	if n.p.IsInfinity() {
		return make([]byte, numBytes*4)
	}

	n.p.MakeAffine(nil)

	xxBytes := new(big.Int).Mod(n.p.x.x, p).Bytes()
	xyBytes := new(big.Int).Mod(n.p.x.y, p).Bytes()
	yxBytes := new(big.Int).Mod(n.p.y.x, p).Bytes()
	yyBytes := new(big.Int).Mod(n.p.y.y, p).Bytes()

	ret := make([]byte, numBytes*4)
	copy(ret[1*numBytes-len(xxBytes):], xxBytes)
	copy(ret[2*numBytes-len(xyBytes):], xyBytes)
	copy(ret[3*numBytes-len(yxBytes):], yxBytes)
	copy(ret[4*numBytes-len(yyBytes):], yyBytes)

	return ret
}

// Unmarshal sets e to the result of converting the output of Marshal back into
// a group element and then returns e.
func (e *G2) Unmarshal(m []byte) (*G2, bool) {
	// Each value is a 256-bit number.
	const numBytes = 256 / 8

	if len(m) != 4*numBytes {
		return nil, false
	}

	if e.p == nil {
		e.p = newTwistPoint(nil)
	}

	e.p.x.x.SetBytes(m[0*numBytes : 1*numBytes])
	e.p.x.y.SetBytes(m[1*numBytes : 2*numBytes])
	e.p.y.x.SetBytes(m[2*numBytes : 3*numBytes])
	e.p.y.y.SetBytes(m[3*numBytes : 4*numBytes])

	if e.p.x.x.Sign() == 0 &&
		e.p.x.y.Sign() == 0 &&
		e.p.y.x.Sign() == 0 &&
		e.p.y.y.Sign() == 0 {
		// This is the point at infinity.
		e.p.y.SetOne()
		e.p.z.SetZero()
		e.p.t.SetZero()
	} else {
		e.p.z.SetOne()
		e.p.t.SetOne()

		if !e.p.IsOnCurve() {
			return nil, false
		}
	}

	return e, true
}

// GT is an abstract cyclic group. The zero value is suitable for use as the
// output of an operation, but cannot be used as an input.
type GT struct {
	p *gfP12
}

func (e *GT) String() string {
	if e.p == nil {
		return "bn256.GT" + newGFp12(nil).String()
	}
	return "bn256.GT" + e.p.String()
}

// ScalarMult sets e to a*k and then returns e.
func (e *GT) ScalarMult(a *GT, k *big.Int) *GT {
	if e.p == nil {
		e.p = newGFp12(nil)
	}
	e.p.Exp(a.p, k, new(bnPool))
	return e
}

// Add sets e to a+b and then returns e.
func (e *GT) Add(a, b *GT) *GT {
	if e.p == nil {
		e.p = newGFp12(nil)
	}
	e.p.Mul(a.p, b.p, new(bnPool))
	return e
}

// Neg sets e to -a and then returns e.
func (e *GT) Neg(a *GT) *GT {
	if e.p == nil {
		e.p = newGFp12(nil)
	}
	e.p.Invert(a.p, new(bnPool))
	return e
}

// Marshal converts n into a byte slice.
func (n *GT) Marshal() []byte {
	n.p.Minimal()

	xxxBytes := n.p.x.x.x.Bytes()
	xxyBytes := n.p.x.x.y.Bytes()
	xyxBytes := n.p.x.y.x.Bytes()
	xyyBytes := n.p.x.y.y.Bytes()
	xzxBytes := n.p.x.z.x.Bytes()
	xzyBytes := n.p.x.z.y.Bytes()
	yxxBytes := n.p.y.x.x.Bytes()
	yxyBytes := n.p.y.x.y.Bytes()
	yyxBytes := n.p.y.y.x.Bytes()
	yyyBytes := n.p.y.y.y.Bytes()
	yzxBytes := n.p.y.z.x.Bytes()
	yzyBytes := n.p.y.z.y.Bytes()

	// Each value is a 256-bit number.
	const numBytes = 256 / 8

	ret := make([]byte, numBytes*12)
	copy(ret[1*numBytes-len(xxxBytes):], xxxBytes)
	copy(ret[2*numBytes-len(xxyBytes):], xxyBytes)
	copy(ret[3*numBytes-len(xyxBytes):], xyxBytes)
	copy(ret[4*numBytes-len(xyyBytes):], xyyBytes)
	copy(ret[5*numBytes-len(xzxBytes):], xzxBytes)
	copy(ret[6*numBytes-len(xzyBytes):], xzyBytes)
	copy(ret[7*numBytes-len(yxxBytes):], yxxBytes)
	copy(ret[8*numBytes-len(yxyBytes):], yxyBytes)
	copy(ret[9*numBytes-len(yyxBytes):], yyxBytes)
	copy(ret[10*numBytes-len(yyyBytes):], yyyBytes)
	copy(ret[11*numBytes-len(yzxBytes):], yzxBytes)
	copy(ret[12*numBytes-len(yzyBytes):], yzyBytes)

	return ret
}

// Unmarshal sets e to the result of converting the output of Marshal back into
// a group element and then returns e.
func (e *GT) Unmarshal(m []byte) (*GT, bool) {
	// Each value is a 256-bit number.
	const numBytes = 256 / 8

	if len(m) != 12*numBytes {
		return nil, false
	}

	if e.p == nil {
		e.p = newGFp12(nil)
	}

	e.p.x.x.x.SetBytes(m[0*numBytes : 1*numBytes])
	e.p.x.x.y.SetBytes(m[1*numBytes : 2*numBytes])
	e.p.x.y.x.SetBytes(m[2*numBytes : 3*numBytes])
	e.p.x.y.y.SetBytes(m[3*numBytes : 4*numBytes])
	e.p.x.z.x.SetBytes(m[4*numBytes : 5*numBytes])
	e.p.x.z.y.SetBytes(m[5*numBytes : 6*numBytes])
	e.p.y.x.x.SetBytes(m[6*numBytes : 7*numBytes])
	e.p.y.x.y.SetBytes(m[7*numBytes : 8*numBytes])
	e.p.y.y.x.SetBytes(m[8*numBytes : 9*numBytes])
	e.p.y.y.y.SetBytes(m[9*numBytes : 10*numBytes])
	e.p.y.z.x.SetBytes(m[10*numBytes : 11*numBytes])
	e.p.y.z.y.SetBytes(m[11*numBytes : 12*numBytes])

	return e, true
}

// Pair calculates an Optimal Ate pairing.
func Pair(g1 *G1, g2 *G2) *GT {
	return &GT{optimalAte(g2.p, g1.p, new(bnPool))}
}

// bnPool implements a tiny cache of *big.Int objects that's used to reduce the
// number of allocations made during processing.
type bnPool struct {
	bns   []*big.Int
	count int
}

func (pool *bnPool) Get() *big.Int {
	if pool == nil {
		return new(big.Int)
	}

	pool.count++
	l := len(pool.bns)
	if l == 0 {
		return new(big.Int)
	}

	bn := pool.bns[l-1]
	pool.bns = pool.bns[:l-1]
	return bn
}

func (pool *bnPool) Put(bn *big.Int) {
	if pool == nil {
		return
	}
	pool.bns = append(pool.bns, bn)
	pool.count--
}

func (pool *bnPool) Count() int {
	return pool.count
}
