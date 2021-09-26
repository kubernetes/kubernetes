// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go

// Package runenames provides rune names from the Unicode Character Database.
// For example, the name for '\u0100' is "LATIN CAPITAL LETTER A WITH MACRON".
//
// See https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt
package runenames

import (
	"sort"
)

// Name returns the name for r.
func Name(r rune) string {
	i := sort.Search(len(entries), func(j int) bool {
		return entries[j].startRune() > r
	})
	if i == 0 {
		return ""
	}
	e := entries[i-1]

	offset := int(r - e.startRune())
	if offset >= e.numRunes() {
		return ""
	}

	if e.direct() {
		o := e.index()
		n := e.len()
		return directData[o : o+n]
	}

	start := int(index[e.index()+offset])
	end := int(index[e.index()+offset+1])
	base1 := e.base() << 16
	base2 := base1
	if start > end {
		base2 += 1 << 16
	}
	return singleData[start+base1 : end+base2]
}

func (e entry) len() int { return e.base() }
