package goabnf

import (
	"fmt"
	"sort"
	"unicode/utf8"
)

// One-symbol lookahead for the GLL engines.
//
// GLL as published guards parsing steps with a selection test test(x, A, γ):
// a sequence γ in a production of A is only explored at input position i when
// the next input symbol x is in FIRST(γ), or γ is nullable and x is in
// FOLLOW(A) (Scott & Johnstone, "GLL parse-tree generation", Science of
// Computer Programming 78(10), 2013, §3.3). It guards the descriptor started
// for each alternate α of A (§4.3), and the call of each nonterminal instance
// A ::= α·Xβ with γ = Xβ (§4.1), so no GSS node is created for a call that
// cannot consume the next symbol.
//
// The guard prunes sub-parses whose continuation cannot consume the next input
// symbol, such as a repetition 1*ALPHA stopping inside a run of letters. This
// keeps the forest proportional to the derivations that can still complete,
// e.g. linear in the run length for 1*ALPHA followed by a delimiter.
//
// A sequence rejected by the guard would have failed on a terminal at the same
// position, so its FIRST (and, when nullable, FOLLOW) terminals feed the
// furthest-failure diagnostics in its place.
//
// Symbols are runes (ABNF terminals are matched on decoded UTF-8), so sets are
// rune intervals plus an end-of-input marker. FIRST/FOLLOW are computed on the
// context-free reading of the slot grammar, ignoring repetition bounds: the
// bounded language is a subset of it, so the sets over-approximate and the test
// never rejects a viable sequence.

type runeRange struct{ lo, hi rune }

// laSet is a lookahead set: rune intervals, the end-of-input marker, and the
// terminals contributing to it (for furthest-failure diagnostics).
type laSet struct {
	ranges []runeRange // sorted, non-overlapping, non-adjacent
	ascii  [2]uint64   // bitmap of ranges below utf8.RuneSelf
	eof    bool
	terms  []ssym
	descs  map[string]bool
}

func (s *laSet) addRange(lo, hi rune) bool {
	if lo > hi {
		return false
	}
	if s.coversRange(lo, hi) {
		return false
	}
	s.ranges = append(s.ranges, runeRange{lo, hi})
	sort.Slice(s.ranges, func(i, j int) bool { return s.ranges[i].lo < s.ranges[j].lo })
	merged := s.ranges[:1]
	for _, r := range s.ranges[1:] {
		last := &merged[len(merged)-1]
		if r.lo <= last.hi+1 {
			if r.hi > last.hi {
				last.hi = r.hi
			}
			continue
		}
		merged = append(merged, r)
	}
	s.ranges = merged
	return true
}

func (s *laSet) coversRange(lo, hi rune) bool {
	for _, r := range s.ranges {
		if r.lo <= lo && hi <= r.hi {
			return true
		}
	}
	return false
}

func (s *laSet) addTerm(t ssym) bool {
	desc := "?"
	if str, ok := t.term.(fmt.Stringer); ok {
		desc = str.String()
	}
	if s.descs == nil {
		s.descs = map[string]bool{}
	}
	if s.descs[desc] {
		return false
	}
	s.descs[desc] = true
	s.terms = append(s.terms, t)
	return true
}

// union adds o to s (optionally including o's end-of-input marker) and
// reports whether s changed.
func (s *laSet) union(o *laSet, withEOF bool) bool {
	changed := false
	for _, r := range o.ranges {
		if s.addRange(r.lo, r.hi) {
			changed = true
		}
	}
	if withEOF && o.eof && !s.eof {
		s.eof = true
		changed = true
	}
	for _, t := range o.terms {
		if s.addTerm(t) {
			changed = true
		}
	}
	return changed
}

func (s *laSet) seal() {
	for _, r := range s.ranges {
		for c := r.lo; c <= r.hi && c < utf8.RuneSelf; c++ {
			if c >= 0 {
				s.ascii[c/64] |= 1 << (c % 64)
			}
		}
	}
}

func (s *laSet) contains(c rune) bool {
	if c >= 0 && c < utf8.RuneSelf {
		return s.ascii[c/64]&(1<<(c%64)) != 0
	}
	i := sort.Search(len(s.ranges), func(i int) bool { return s.ranges[i].hi >= c })
	return i < len(s.ranges) && s.ranges[i].lo <= c
}

// termFirst returns the FIRST set of a terminal and whether it matches the
// empty string, mirroring matchTerm.
func termFirst(t ssym) (laSet, bool) {
	var s laSet
	if t.kind == symEps {
		return s, true
	}
	switch v := t.term.(type) {
	case ElemCharVal:
		if len(v.Values) == 0 {
			return s, true
		}
		r := v.Values[0]
		if v.Sensitive {
			s.addRange(r, r)
		} else {
			lo := runeMin(r)
			s.addRange(lo, lo)
			up := runeMax(lo)
			s.addRange(up, up)
		}

	case ElemNumVal:
		switch v.Status {
		case StatRange:
			s.addRange(numvalToRune(v.Elems[0], v.Base), numvalToRune(v.Elems[1], v.Base))

		case StatSeries:
			if len(v.Elems) == 0 {
				return s, true
			}
			if r := numvalToRune(v.Elems[0], v.Base); utf8.ValidRune(r) {
				s.addRange(r, r)
			}
		}
	}
	// A terminal that never matches (prose-val, out-of-range series) has an
	// empty rune set but is still an attempted terminal for diagnostics.
	s.addTerm(t)
	return s, false
}

// lookahead holds the selection sets FIRST(γ) ∪ (FOLLOW(A) if γ is nullable)
// of the slot grammar: sel for every alternate γ = α of A, call for every
// nonterminal instance A ::= α·Xβ with γ = Xβ.
type lookahead struct {
	sel  [][]laSet   // [nt][alt]
	call [][][]laSet // [nt][alt][dot], set only where the symbol at dot is a nonterminal
}

func computeLookahead(sg *slotGrammar) *lookahead {
	n := len(sg.nts)
	nullable := make([]bool, n)
	first := make([]laSet, n)
	follow := make([]laSet, n)

	symNullable := func(s ssym) bool {
		switch s.kind {
		case symNonterm:
			return nullable[s.nt]

		default:
			_, null := termFirst(s)
			return null
		}
	}
	// firstSeq returns FIRST(seq) and whether seq is nullable.
	firstSeq := func(seq []ssym) (laSet, bool) {
		var out laSet
		for _, s := range seq {
			if s.kind == symNonterm {
				out.union(&first[s.nt], false)
			} else {
				f, _ := termFirst(s)
				out.union(&f, false)
			}
			if !symNullable(s) {
				return out, false
			}
		}
		return out, true
	}

	for changed := true; changed; {
		changed = false
		for a, nt := range sg.nts {
			for _, alt := range nt.alts {
				f, null := firstSeq(alt)
				if first[a].union(&f, false) {
					changed = true
				}
				if null && !nullable[a] {
					nullable[a] = true
					changed = true
				}
			}
		}
	}

	follow[sg.start].eof = true
	for changed := true; changed; {
		changed = false
		for a, nt := range sg.nts {
			for _, alt := range nt.alts {
				for k, s := range alt {
					if s.kind != symNonterm {
						continue
					}
					f, null := firstSeq(alt[k+1:])
					if follow[s.nt].union(&f, false) {
						changed = true
					}
					if null && follow[s.nt].union(&follow[a], true) {
						changed = true
					}
				}
			}
		}
	}

	selection := func(a int, seq []ssym) laSet {
		f, null := firstSeq(seq)
		if null {
			f.union(&follow[a], true)
		}
		f.seal()
		return f
	}
	la := &lookahead{sel: make([][]laSet, n), call: make([][][]laSet, n)}
	for a, nt := range sg.nts {
		la.sel[a] = make([]laSet, len(nt.alts))
		la.call[a] = make([][]laSet, len(nt.alts))
		for ai, alt := range nt.alts {
			la.sel[a][ai] = selection(a, alt)
			la.call[a][ai] = make([]laSet, len(alt))
			for k, sym := range alt {
				if sym.kind == symNonterm {
					la.call[a][ai][k] = selection(a, alt[k:])
				}
			}
		}
	}
	return la
}

// alt returns the selection set guarding alternate alt of nonterminal nt.
func (la *lookahead) alt(nt, alt int) *laSet { return &la.sel[nt][alt] }

// callAt returns the selection set guarding the nonterminal call at slot L.
func (la *lookahead) callAt(L slot) *laSet { return &la.call[L.nt][L.alt][L.dot] }

// admits reports whether the next input symbol at i is in s, i.e. whether the
// guarded sequence may derive a prefix of input[i:] in some context.
func (s *laSet) admits(input []byte, i int) bool {
	if i >= len(input) {
		return s.eof
	}
	r, size := utf8.DecodeRune(input[i:])
	if r == utf8.RuneError && size == 1 {
		// No terminal matches an invalid UTF-8 sequence (see matchTerm).
		return false
	}
	return s.contains(r)
}
