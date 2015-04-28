// Copyright (c) 2011 jnml. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Let QRN be the number of quadratic residues of N.  Let Q be QRN/N.  From a
// sorted list of primorial products < 2^32 find "record breakers".  "Record
// breaker" is N with new lowest Q.
//
// There are only 49 "record breakers" < 2^32.
//
// To run the example $ go run main.go
package main

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/cznic/mathutil"
	"github.com/cznic/sortutil"
)

func main() {
	pp := mathutil.PrimorialProductsUint32(0, math.MaxUint32, 32)
	sort.Sort(sortutil.Uint32Slice(pp))
	var bestN, bestD uint32 = 1, 1
	order, checks := 0, 0
	var ixDirty uint32
	m := make([]byte, math.MaxUint32>>3)
	for _, n := range pp {
		for i := range m[:ixDirty+1] {
			m[i] = 0
		}
		ixDirty = 0
		checks++
		limit0 := mathutil.QScaleUint32(n, bestN, bestD)
		if limit0 > math.MaxUint32 {
			panic(0)
		}
		limit := uint32(limit0)
		n64 := uint64(n)
		hi := n64 >> 1
		hits := uint32(0)
		check := true
		fmt.Printf("\r%10d %d/%d", n, checks, len(pp))
		t0 := time.Now()
		for i := uint64(0); i < hi; i++ {
			sq := uint32(i * i % n64)
			ix := sq >> 3
			msk := byte(1 << (sq & 7))
			if m[ix]&msk == 0 {
				hits++
				if hits >= limit {
					check = false
					break
				}
			}
			m[ix] |= msk
			if ix > ixDirty {
				ixDirty = ix
			}
		}

		adjPrime := ".." // Composite before
		if mathutil.IsPrime(n - 1) {
			adjPrime = "P." // Prime before
		}
		switch mathutil.IsPrime(n + 1) {
		case true:
			adjPrime += "P" // Prime after
		case false:
			adjPrime += "." // Composite after
		}

		if check && mathutil.QCmpUint32(hits, n, bestN, bestD) < 0 {
			order++
			d := time.Since(t0)
			bestN, bestD = hits, n
			q := float64(hits) / float64(n)
			fmt.Printf(
				"\r%2s #%03d %d %d %.2f %.2E %s %s %v\n",
				adjPrime, order, n, hits,
				1/q, q, d, time.Now().Format("15:04:05"), mathutil.FactorInt(n),
			)
		}
	}
}
