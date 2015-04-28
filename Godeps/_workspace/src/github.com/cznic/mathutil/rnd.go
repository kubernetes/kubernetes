// Copyright (c) 2014 The mathutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mathutil

import (
	"fmt"
	"math"
	"math/big"
)

// FC32 is a full cycle PRNG covering the 32 bit signed integer range.
// In contrast to full cycle generators shown at e.g. http://en.wikipedia.org/wiki/Full_cycle,
// this code doesn't produce values at constant delta (mod cycle length).
// The 32 bit limit is per this implementation, the algorithm used has no intrinsic limit on the cycle size.
// Properties include:
//	- Adjustable limits on creation (hi, lo).
//	- Positionable/randomly accessible (Pos, Seek).
//	- Repeatable (deterministic).
//	- Can run forward or backward (Next, Prev).
//	- For a billion numbers cycle the Next/Prev PRN can be produced in cca 100-150ns.
//	  That's like 5-10 times slower compared to PRNs generated using the (non FC) rand package.
type FC32 struct {
	cycle   int64     // On average: 3 * delta / 2, (HQ: 2 * delta)
	delta   int64     // hi - lo
	factors [][]int64 // This trades some space for hopefully a bit of speed (multiple adding vs multiplying).
	lo      int
	mods    []int   // pos % set
	pos     int64   // Within cycle.
	primes  []int64 // Ordered. ∏ primes == cycle.
	set     []int64 // Reordered primes (magnitude order bases) according to seed.
}

// NewFC32 returns a newly created FC32 adjusted for the closed interval [lo, hi] or an Error if any.
// If hq == true then trade some generation time for improved (pseudo)randomness.
func NewFC32(lo, hi int, hq bool) (r *FC32, err error) {
	if lo > hi {
		return nil, fmt.Errorf("invalid range %d > %d", lo, hi)
	}

	if uint64(hi)-uint64(lo) > math.MaxUint32 {
		return nil, fmt.Errorf("range out of int32 limits %d, %d", lo, hi)
	}

	delta := int64(hi) - int64(lo)
	// Find the primorial covering whole delta
	n, set, p := int64(1), []int64{}, uint32(2)
	if hq {
		p++
	}
	for {
		set = append(set, int64(p))
		n *= int64(p)
		if n > delta {
			break
		}
		p, _ = NextPrime(p)
	}

	// Adjust the set so n ∊ [delta, 2 * delta] (HQ: [delta, 3 * delta])
	// while keeping the cardinality of the set (correlates with the statistic "randomness quality")
	// at max, i.e. discard atmost one member.
	i := -1 // no candidate prime
	if n > 2*(delta+1) {
		for j, p := range set {
			q := n / p
			if q < delta+1 {
				break
			}

			i = j // mark the highest candidate prime set index
		}
	}
	if i >= 0 { // shrink the inner cycle
		n = n / set[i]
		set = delete(set, i)
	}
	r = &FC32{
		cycle:   n,
		delta:   delta,
		factors: make([][]int64, len(set)),
		lo:      lo,
		mods:    make([]int, len(set)),
		primes:  set,
	}
	r.Seed(1) // the default seed should be always non zero
	return
}

// Cycle reports the length of the inner FCPRNG cycle.
// Cycle is atmost the double (HQ: triple) of the generator period (hi - lo + 1).
func (r *FC32) Cycle() int64 {
	return r.cycle
}

// Next returns the first PRN after Pos.
func (r *FC32) Next() int {
	return r.step(1)
}

// Pos reports the current position within the inner cycle.
func (r *FC32) Pos() int64 {
	return r.pos
}

// Prev return the first PRN before Pos.
func (r *FC32) Prev() int {
	return r.step(-1)
}

// Seed uses the provided seed value to initialize the generator to a deterministic state.
// A zero seed produces a "canonical" generator with worse randomness than for most non zero seeds.
// Still, the FC property holds for any seed value.
func (r *FC32) Seed(seed int64) {
	u := uint64(seed)
	r.set = mix(r.primes, &u)
	n := int64(1)
	for i, p := range r.set {
		k := make([]int64, p)
		v := int64(0)
		for j := range k {
			k[j] = v
			v += n
		}
		n *= p
		r.factors[i] = mix(k, &u)
	}
}

// Seek sets Pos to |pos| % Cycle.
func (r *FC32) Seek(pos int64) { //vet:ignore
	if pos < 0 {
		pos = -pos
	}
	pos %= r.cycle
	r.pos = pos
	for i, p := range r.set {
		r.mods[i] = int(pos % p)
	}
}

func (r *FC32) step(dir int) int {
	for { // avg loops per step: 3/2 (HQ: 2)
		y := int64(0)
		pos := r.pos
		pos += int64(dir)
		switch {
		case pos < 0:
			pos = r.cycle - 1
		case pos >= r.cycle:
			pos = 0
		}
		r.pos = pos
		for i, mod := range r.mods {
			mod += dir
			p := int(r.set[i])
			switch {
			case mod < 0:
				mod = p - 1
			case mod >= p:
				mod = 0
			}
			r.mods[i] = mod
			y += r.factors[i][mod]
		}
		if y <= r.delta {
			return int(y) + r.lo
		}
	}
}

func delete(set []int64, i int) (y []int64) {
	for j, v := range set {
		if j != i {
			y = append(y, v)
		}
	}
	return
}

func mix(set []int64, seed *uint64) (y []int64) {
	for len(set) != 0 {
		*seed = rol(*seed)
		i := int(*seed % uint64(len(set)))
		y = append(y, set[i])
		set = delete(set, i)
	}
	return
}

func rol(u uint64) (y uint64) {
	y = u << 1
	if int64(u) < 0 {
		y |= 1
	}
	return
}

// FCBig is a full cycle PRNG covering ranges outside of the int32 limits.
// For more info see the FC32 docs.
// Next/Prev PRN on a 1e15 cycle can be produced in about 2 µsec.
type FCBig struct {
	cycle   *big.Int     // On average: 3 * delta / 2, (HQ: 2 * delta)
	delta   *big.Int     // hi - lo
	factors [][]*big.Int // This trades some space for hopefully a bit of speed (multiple adding vs multiplying).
	lo      *big.Int
	mods    []int    // pos % set
	pos     *big.Int // Within cycle.
	primes  []int64  // Ordered. ∏ primes == cycle.
	set     []int64  // Reordered primes (magnitude order bases) according to seed.
}

// NewFCBig returns a newly created FCBig adjusted for the closed interval [lo, hi] or an Error if any.
// If hq == true then trade some generation time for improved (pseudo)randomness.
func NewFCBig(lo, hi *big.Int, hq bool) (r *FCBig, err error) {
	if lo.Cmp(hi) > 0 {
		return nil, fmt.Errorf("invalid range %d > %d", lo, hi)
	}

	delta := big.NewInt(0)
	delta.Add(delta, hi).Sub(delta, lo)

	// Find the primorial covering whole delta
	n, set, pp, p := big.NewInt(1), []int64{}, big.NewInt(0), uint32(2)
	if hq {
		p++
	}
	for {
		set = append(set, int64(p))
		pp.SetInt64(int64(p))
		n.Mul(n, pp)
		if n.Cmp(delta) > 0 {
			break
		}
		p, _ = NextPrime(p)
	}

	// Adjust the set so n ∊ [delta, 2 * delta] (HQ: [delta, 3 * delta])
	// while keeping the cardinality of the set (correlates with the statistic "randomness quality")
	// at max, i.e. discard atmost one member.
	dd1 := big.NewInt(1)
	dd1.Add(dd1, delta)
	dd2 := big.NewInt(0)
	dd2.Lsh(dd1, 1)
	i := -1 // no candidate prime
	if n.Cmp(dd2) > 0 {
		q := big.NewInt(0)
		for j, p := range set {
			pp.SetInt64(p)
			q.Set(n)
			q.Div(q, pp)
			if q.Cmp(dd1) < 0 {
				break
			}

			i = j // mark the highest candidate prime set index
		}
	}
	if i >= 0 { // shrink the inner cycle
		pp.SetInt64(set[i])
		n.Div(n, pp)
		set = delete(set, i)
	}
	r = &FCBig{
		cycle:   n,
		delta:   delta,
		factors: make([][]*big.Int, len(set)),
		lo:      lo,
		mods:    make([]int, len(set)),
		pos:     big.NewInt(0),
		primes:  set,
	}
	r.Seed(1) // the default seed should be always non zero
	return
}

// Cycle reports the length of the inner FCPRNG cycle.
// Cycle is atmost the double (HQ: triple) of the generator period (hi - lo + 1).
func (r *FCBig) Cycle() *big.Int {
	return r.cycle
}

// Next returns the first PRN after Pos.
func (r *FCBig) Next() *big.Int {
	return r.step(1)
}

// Pos reports the current position within the inner cycle.
func (r *FCBig) Pos() *big.Int {
	return r.pos
}

// Prev return the first PRN before Pos.
func (r *FCBig) Prev() *big.Int {
	return r.step(-1)
}

// Seed uses the provided seed value to initialize the generator to a deterministic state.
// A zero seed produces a "canonical" generator with worse randomness than for most non zero seeds.
// Still, the FC property holds for any seed value.
func (r *FCBig) Seed(seed int64) {
	u := uint64(seed)
	r.set = mix(r.primes, &u)
	n := big.NewInt(1)
	v := big.NewInt(0)
	pp := big.NewInt(0)
	for i, p := range r.set {
		k := make([]*big.Int, p)
		v.SetInt64(0)
		for j := range k {
			k[j] = big.NewInt(0)
			k[j].Set(v)
			v.Add(v, n)
		}
		pp.SetInt64(p)
		n.Mul(n, pp)
		r.factors[i] = mixBig(k, &u)
	}
}

// Seek sets Pos to |pos| % Cycle.
func (r *FCBig) Seek(pos *big.Int) {
	r.pos.Set(pos)
	r.pos.Abs(r.pos)
	r.pos.Mod(r.pos, r.cycle)
	mod := big.NewInt(0)
	pp := big.NewInt(0)
	for i, p := range r.set {
		pp.SetInt64(p)
		r.mods[i] = int(mod.Mod(r.pos, pp).Int64())
	}
}

func (r *FCBig) step(dir int) (y *big.Int) {
	y = big.NewInt(0)
	d := big.NewInt(int64(dir))
	for { // avg loops per step: 3/2 (HQ: 2)
		r.pos.Add(r.pos, d)
		switch {
		case r.pos.Sign() < 0:
			r.pos.Add(r.pos, r.cycle)
		case r.pos.Cmp(r.cycle) >= 0:
			r.pos.SetInt64(0)
		}
		for i, mod := range r.mods {
			mod += dir
			p := int(r.set[i])
			switch {
			case mod < 0:
				mod = p - 1
			case mod >= p:
				mod = 0
			}
			r.mods[i] = mod
			y.Add(y, r.factors[i][mod])
		}
		if y.Cmp(r.delta) <= 0 {
			y.Add(y, r.lo)
			return
		}
		y.SetInt64(0)
	}
}

func deleteBig(set []*big.Int, i int) (y []*big.Int) {
	for j, v := range set {
		if j != i {
			y = append(y, v)
		}
	}
	return
}

func mixBig(set []*big.Int, seed *uint64) (y []*big.Int) {
	for len(set) != 0 {
		*seed = rol(*seed)
		i := int(*seed % uint64(len(set)))
		y = append(y, set[i])
		set = deleteBig(set, i)
	}
	return
}
