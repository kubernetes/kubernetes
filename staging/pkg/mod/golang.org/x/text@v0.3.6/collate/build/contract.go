// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"

	"golang.org/x/text/internal/colltab"
)

// This file contains code for detecting contractions and generating
// the necessary tables.
// Any Unicode Collation Algorithm (UCA) table entry that has more than
// one rune one the left-hand side is called a contraction.
// See https://www.unicode.org/reports/tr10/#Contractions for more details.
//
// We define the following terms:
//   initial:     a rune that appears as the first rune in a contraction.
//   suffix:      a sequence of runes succeeding the initial rune
//                in a given contraction.
//   non-initial: a rune that appears in a suffix.
//
// A rune may be both an initial and a non-initial and may be so in
// many contractions.  An initial may typically also appear by itself.
// In case of ambiguities, the UCA requires we match the longest
// contraction.
//
// Many contraction rules share the same set of possible suffixes.
// We store sets of suffixes in a trie that associates an index with
// each suffix in the set.  This index can be used to look up a
// collation element associated with the (starter rune, suffix) pair.
//
// The trie is defined on a UTF-8 byte sequence.
// The overall trie is represented as an array of ctEntries.  Each node of the trie
// is represented as a subsequence of ctEntries, where each entry corresponds to
// a possible match of a next character in the search string.  An entry
// also includes the length and offset to the next sequence of entries
// to check in case of a match.

const (
	final   = 0
	noIndex = 0xFF
)

// ctEntry associates to a matching byte an offset and/or next sequence of
// bytes to check. A ctEntry c is called final if a match means that the
// longest suffix has been found.  An entry c is final if c.N == 0.
// A single final entry can match a range of characters to an offset.
// A non-final entry always matches a single byte. Note that a non-final
// entry might still resemble a completed suffix.
// Examples:
// The suffix strings "ab" and "ac" can be represented as:
// []ctEntry{
//     {'a', 1, 1, noIndex},  // 'a' by itself does not match, so i is 0xFF.
//     {'b', 'c', 0, 1},   // "ab" -> 1, "ac" -> 2
// }
//
// The suffix strings "ab", "abc", "abd", and "abcd" can be represented as:
// []ctEntry{
//     {'a', 1, 1, noIndex}, // 'a' must be followed by 'b'.
//     {'b', 1, 2, 1},    // "ab" -> 1, may be followed by 'c' or 'd'.
//     {'d', 'd', final, 3},  // "abd" -> 3
//     {'c', 4, 1, 2},    // "abc" -> 2, may be followed by 'd'.
//     {'d', 'd', final, 4},  // "abcd" -> 4
// }
// See genStateTests in contract_test.go for more examples.
type ctEntry struct {
	L uint8 // non-final: byte value to match; final: lowest match in range.
	H uint8 // non-final: relative index to next block; final: highest match in range.
	N uint8 // non-final: length of next block; final: final
	I uint8 // result offset. Will be noIndex if more bytes are needed to complete.
}

// contractTrieSet holds a set of contraction tries. The tries are stored
// consecutively in the entry field.
type contractTrieSet []struct{ l, h, n, i uint8 }

// ctHandle is used to identify a trie in the trie set, consisting in an offset
// in the array and the size of the first node.
type ctHandle struct {
	index, n int
}

// appendTrie adds a new trie for the given suffixes to the trie set and returns
// a handle to it.  The handle will be invalid on error.
func appendTrie(ct *colltab.ContractTrieSet, suffixes []string) (ctHandle, error) {
	es := make([]stridx, len(suffixes))
	for i, s := range suffixes {
		es[i].str = s
	}
	sort.Sort(offsetSort(es))
	for i := range es {
		es[i].index = i + 1
	}
	sort.Sort(genidxSort(es))
	i := len(*ct)
	n, err := genStates(ct, es)
	if err != nil {
		*ct = (*ct)[:i]
		return ctHandle{}, err
	}
	return ctHandle{i, n}, nil
}

// genStates generates ctEntries for a given suffix set and returns
// the number of entries for the first node.
func genStates(ct *colltab.ContractTrieSet, sis []stridx) (int, error) {
	if len(sis) == 0 {
		return 0, fmt.Errorf("genStates: list of suffices must be non-empty")
	}
	start := len(*ct)
	// create entries for differing first bytes.
	for _, si := range sis {
		s := si.str
		if len(s) == 0 {
			continue
		}
		added := false
		c := s[0]
		if len(s) > 1 {
			for j := len(*ct) - 1; j >= start; j-- {
				if (*ct)[j].L == c {
					added = true
					break
				}
			}
			if !added {
				*ct = append(*ct, ctEntry{L: c, I: noIndex})
			}
		} else {
			for j := len(*ct) - 1; j >= start; j-- {
				// Update the offset for longer suffixes with the same byte.
				if (*ct)[j].L == c {
					(*ct)[j].I = uint8(si.index)
					added = true
				}
				// Extend range of final ctEntry, if possible.
				if (*ct)[j].H+1 == c {
					(*ct)[j].H = c
					added = true
				}
			}
			if !added {
				*ct = append(*ct, ctEntry{L: c, H: c, N: final, I: uint8(si.index)})
			}
		}
	}
	n := len(*ct) - start
	// Append nodes for the remainder of the suffixes for each ctEntry.
	sp := 0
	for i, end := start, len(*ct); i < end; i++ {
		fe := (*ct)[i]
		if fe.H == 0 { // uninitialized non-final
			ln := len(*ct) - start - n
			if ln > 0xFF {
				return 0, fmt.Errorf("genStates: relative block offset too large: %d > 255", ln)
			}
			fe.H = uint8(ln)
			// Find first non-final strings with same byte as current entry.
			for ; sis[sp].str[0] != fe.L; sp++ {
			}
			se := sp + 1
			for ; se < len(sis) && len(sis[se].str) > 1 && sis[se].str[0] == fe.L; se++ {
			}
			sl := sis[sp:se]
			sp = se
			for i, si := range sl {
				sl[i].str = si.str[1:]
			}
			nn, err := genStates(ct, sl)
			if err != nil {
				return 0, err
			}
			fe.N = uint8(nn)
			(*ct)[i] = fe
		}
	}
	sort.Sort(entrySort((*ct)[start : start+n]))
	return n, nil
}

// There may be both a final and non-final entry for a byte if the byte
// is implied in a range of matches in the final entry.
// We need to ensure that the non-final entry comes first in that case.
type entrySort colltab.ContractTrieSet

func (fe entrySort) Len() int      { return len(fe) }
func (fe entrySort) Swap(i, j int) { fe[i], fe[j] = fe[j], fe[i] }
func (fe entrySort) Less(i, j int) bool {
	return fe[i].L > fe[j].L
}

// stridx is used for sorting suffixes and their associated offsets.
type stridx struct {
	str   string
	index int
}

// For computing the offsets, we first sort by size, and then by string.
// This ensures that strings that only differ in the last byte by 1
// are sorted consecutively in increasing order such that they can
// be packed as a range in a final ctEntry.
type offsetSort []stridx

func (si offsetSort) Len() int      { return len(si) }
func (si offsetSort) Swap(i, j int) { si[i], si[j] = si[j], si[i] }
func (si offsetSort) Less(i, j int) bool {
	if len(si[i].str) != len(si[j].str) {
		return len(si[i].str) > len(si[j].str)
	}
	return si[i].str < si[j].str
}

// For indexing, we want to ensure that strings are sorted in string order, where
// for strings with the same prefix, we put longer strings before shorter ones.
type genidxSort []stridx

func (si genidxSort) Len() int      { return len(si) }
func (si genidxSort) Swap(i, j int) { si[i], si[j] = si[j], si[i] }
func (si genidxSort) Less(i, j int) bool {
	if strings.HasPrefix(si[j].str, si[i].str) {
		return false
	}
	if strings.HasPrefix(si[i].str, si[j].str) {
		return true
	}
	return si[i].str < si[j].str
}

// lookup matches the longest suffix in str and returns the associated offset
// and the number of bytes consumed.
func lookup(ct *colltab.ContractTrieSet, h ctHandle, str []byte) (index, ns int) {
	states := (*ct)[h.index:]
	p := 0
	n := h.n
	for i := 0; i < n && p < len(str); {
		e := states[i]
		c := str[p]
		if c >= e.L {
			if e.L == c {
				p++
				if e.I != noIndex {
					index, ns = int(e.I), p
				}
				if e.N != final {
					// set to new state
					i, states, n = 0, states[int(e.H)+n:], int(e.N)
				} else {
					return
				}
				continue
			} else if e.N == final && c <= e.H {
				p++
				return int(c-e.L) + int(e.I), p
			}
		}
		i++
	}
	return
}

// print writes the contractTrieSet t as compilable Go code to w. It returns
// the total number of bytes written and the size of the resulting data structure in bytes.
func print(t *colltab.ContractTrieSet, w io.Writer, name string) (n, size int, err error) {
	update3 := func(nn, sz int, e error) {
		n += nn
		if err == nil {
			err = e
		}
		size += sz
	}
	update2 := func(nn int, e error) { update3(nn, 0, e) }

	update3(printArray(*t, w, name))
	update2(fmt.Fprintf(w, "var %sContractTrieSet = ", name))
	update3(printStruct(*t, w, name))
	update2(fmt.Fprintln(w))
	return
}

func printArray(ct colltab.ContractTrieSet, w io.Writer, name string) (n, size int, err error) {
	p := func(f string, a ...interface{}) {
		nn, e := fmt.Fprintf(w, f, a...)
		n += nn
		if err == nil {
			err = e
		}
	}
	size = len(ct) * 4
	p("// %sCTEntries: %d entries, %d bytes\n", name, len(ct), size)
	p("var %sCTEntries = [%d]struct{L,H,N,I uint8}{\n", name, len(ct))
	for _, fe := range ct {
		p("\t{0x%X, 0x%X, %d, %d},\n", fe.L, fe.H, fe.N, fe.I)
	}
	p("}\n")
	return
}

func printStruct(ct colltab.ContractTrieSet, w io.Writer, name string) (n, size int, err error) {
	n, err = fmt.Fprintf(w, "colltab.ContractTrieSet( %sCTEntries[:] )", name)
	size = int(reflect.TypeOf(ct).Size())
	return
}
