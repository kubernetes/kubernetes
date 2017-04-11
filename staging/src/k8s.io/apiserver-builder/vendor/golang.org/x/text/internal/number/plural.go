// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import "golang.org/x/text/internal/format/plural"

type pluralRules struct {
	rules          []pluralCheck
	index          []byte
	langToIndex    []byte
	inclusionMasks []uint64
}

var (
	ordinalData = pluralRules{
		ordinalRules,
		ordinalIndex,
		ordinalLangToIndex,
		ordinalInclusionMasks[:],
	}
	cardinalData = pluralRules{
		cardinalRules,
		cardinalIndex,
		cardinalLangToIndex,
		cardinalInclusionMasks[:],
	}
)

// See gen_plural.go for an explanation of the algorithm.

func matchPlural(p *pluralRules, index int, n, f, v int) plural.Form {
	nMask := p.inclusionMasks[n%maxMod]
	// Compute the fMask inline in the rules below, as it is relatively rare.
	// fMask := p.inclusionMasks[f%maxMod]
	vMask := p.inclusionMasks[v%maxMod]

	// Do the matching
	offset := p.langToIndex[index]
	rules := p.rules[p.index[offset]:p.index[offset+1]]
	for i := 0; i < len(rules); i++ {
		rule := rules[i]
		setBit := uint64(1 << rule.setID)
		var skip bool
		switch op := opID(rule.cat >> opShift); op {
		case opI: // i = x
			skip = n >= numN || nMask&setBit == 0

		case opI | opNotEqual: // i != x
			skip = n < numN && nMask&setBit != 0

		case opI | opMod: // i % m = x
			skip = nMask&setBit == 0

		case opI | opMod | opNotEqual: // i % m != x
			skip = nMask&setBit != 0

		case opN: // n = x
			skip = f != 0 || n >= numN || nMask&setBit == 0

		case opN | opNotEqual: // n != x
			skip = f == 0 && n < numN && nMask&setBit != 0

		case opN | opMod: // n % m = x
			skip = f != 0 || nMask&setBit == 0

		case opN | opMod | opNotEqual: // n % m != x
			skip = f == 0 && nMask&setBit != 0

		case opF: // f = x
			skip = f >= numN || p.inclusionMasks[f%maxMod]&setBit == 0

		case opF | opNotEqual: // f != x
			skip = f < numN && p.inclusionMasks[f%maxMod]&setBit != 0

		case opF | opMod: // f % m = x
			skip = p.inclusionMasks[f%maxMod]&setBit == 0

		case opF | opMod | opNotEqual: // f % m != x
			skip = p.inclusionMasks[f%maxMod]&setBit != 0

		case opV: // v = x
			skip = v < numN && vMask&setBit == 0

		case opV | opNotEqual: // v != x
			skip = v < numN && vMask&setBit != 0

		case opW: // w == 0
			skip = f != 0

		case opW | opNotEqual: // w != 0
			skip = f == 0

		// Hard-wired rules that cannot be handled by our algorithm.

		case opBretonM:
			skip = f != 0 || n == 0 || n%1000000 != 0

		case opAzerbaijan00s:
			// 100,200,300,400,500,600,700,800,900
			skip = n == 0 || n >= 1000 || n%100 != 0

		case opItalian800:
			skip = (f != 0 || n >= numN || nMask&setBit == 0) && n != 800
		}
		if skip {
			// advance over AND entries.
			for ; i < len(rules) && rules[i].cat&formMask == andNext; i++ {
			}
			continue
		}
		// return if we have a final entry.
		if cat := rule.cat & formMask; cat != andNext {
			return plural.Form(cat)
		}
	}
	return plural.Other
}
