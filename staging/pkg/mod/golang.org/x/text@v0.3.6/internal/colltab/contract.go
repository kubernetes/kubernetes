// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colltab

import "unicode/utf8"

// For a description of ContractTrieSet, see text/collate/build/contract.go.

type ContractTrieSet []struct{ L, H, N, I uint8 }

// ctScanner is used to match a trie to an input sequence.
// A contraction may match a non-contiguous sequence of bytes in an input string.
// For example, if there is a contraction for <a, combining_ring>, it should match
// the sequence <a, combining_cedilla, combining_ring>, as combining_cedilla does
// not block combining_ring.
// ctScanner does not automatically skip over non-blocking non-starters, but rather
// retains the state of the last match and leaves it up to the user to continue
// the match at the appropriate points.
type ctScanner struct {
	states ContractTrieSet
	s      []byte
	n      int
	index  int
	pindex int
	done   bool
}

type ctScannerString struct {
	states ContractTrieSet
	s      string
	n      int
	index  int
	pindex int
	done   bool
}

func (t ContractTrieSet) scanner(index, n int, b []byte) ctScanner {
	return ctScanner{s: b, states: t[index:], n: n}
}

func (t ContractTrieSet) scannerString(index, n int, str string) ctScannerString {
	return ctScannerString{s: str, states: t[index:], n: n}
}

// result returns the offset i and bytes consumed p so far.  If no suffix
// matched, i and p will be 0.
func (s *ctScanner) result() (i, p int) {
	return s.index, s.pindex
}

func (s *ctScannerString) result() (i, p int) {
	return s.index, s.pindex
}

const (
	final   = 0
	noIndex = 0xFF
)

// scan matches the longest suffix at the current location in the input
// and returns the number of bytes consumed.
func (s *ctScanner) scan(p int) int {
	pr := p // the p at the rune start
	str := s.s
	states, n := s.states, s.n
	for i := 0; i < n && p < len(str); {
		e := states[i]
		c := str[p]
		// TODO: a significant number of contractions are of a form that
		// cannot match discontiguous UTF-8 in a normalized string. We could let
		// a negative value of e.n mean that we can set s.done = true and avoid
		// the need for additional matches.
		if c >= e.L {
			if e.L == c {
				p++
				if e.I != noIndex {
					s.index = int(e.I)
					s.pindex = p
				}
				if e.N != final {
					i, states, n = 0, states[int(e.H)+n:], int(e.N)
					if p >= len(str) || utf8.RuneStart(str[p]) {
						s.states, s.n, pr = states, n, p
					}
				} else {
					s.done = true
					return p
				}
				continue
			} else if e.N == final && c <= e.H {
				p++
				s.done = true
				s.index = int(c-e.L) + int(e.I)
				s.pindex = p
				return p
			}
		}
		i++
	}
	return pr
}

// scan is a verbatim copy of ctScanner.scan.
func (s *ctScannerString) scan(p int) int {
	pr := p // the p at the rune start
	str := s.s
	states, n := s.states, s.n
	for i := 0; i < n && p < len(str); {
		e := states[i]
		c := str[p]
		// TODO: a significant number of contractions are of a form that
		// cannot match discontiguous UTF-8 in a normalized string. We could let
		// a negative value of e.n mean that we can set s.done = true and avoid
		// the need for additional matches.
		if c >= e.L {
			if e.L == c {
				p++
				if e.I != noIndex {
					s.index = int(e.I)
					s.pindex = p
				}
				if e.N != final {
					i, states, n = 0, states[int(e.H)+n:], int(e.N)
					if p >= len(str) || utf8.RuneStart(str[p]) {
						s.states, s.n, pr = states, n, p
					}
				} else {
					s.done = true
					return p
				}
				continue
			} else if e.N == final && c <= e.H {
				p++
				s.done = true
				s.index = int(c-e.L) + int(e.I)
				s.pindex = p
				return p
			}
		}
		i++
	}
	return pr
}
