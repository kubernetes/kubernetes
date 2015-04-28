// Copyright (c) jnml. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Factor Finder - searches for Mersenne number factors of one specific special
// form.
package main

import (
	"flag"
	"fmt"
	"math/big"
	"runtime"
	"time"

	"github.com/cznic/mathutil"
)

const (
	pp  = 1
	pp2 = 10
)

var (
	_1 = big.NewInt(1)
	_2 = big.NewInt(2)
)

func main() {
	runtime.GOMAXPROCS(2)
	oClass := flag.Uint64("c", 2, `factor "class" number`)
	oDuration := flag.Duration("d", time.Second, "duration to spend on one class")
	flag.Parse()
	class := *oClass
	for class&1 != 0 {
		class >>= 1
	}
	class = mathutil.MaxUint64(class, 2)

	for {
		c := time.After(*oDuration)
		factor := big.NewInt(0)
		factor.SetUint64(class)
		exp := big.NewInt(0)
	oneClass:
		for {
			select {
			case <-c:
				break oneClass
			default:
			}

			exp.Set(factor)
			factor.Lsh(factor, 1)
			factor.Add(factor, _1)
			if !factor.ProbablyPrime(pp) {
				continue
			}

			if !exp.ProbablyPrime(pp) {
				continue
			}

			if mathutil.ModPowBigInt(_2, exp, factor).Cmp(_1) != 0 {
				continue
			}

			if !factor.ProbablyPrime(pp2) {
				continue
			}

			if !exp.ProbablyPrime(pp2) {
				continue
			}

			fmt.Printf("%d: %s | M%s (%d bits)\n", class, factor, exp, factor.BitLen())
		}

		class += 2
	}
}
