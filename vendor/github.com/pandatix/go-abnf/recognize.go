package goabnf

import (
	"strconv"
	"strings"
	"unicode/utf8"
)

// recognize.go holds the engine behind (*Grammar).IsValid.
//
// Validity only needs the SET of input positions each element can reach, never
// the (possibly exponential) set of distinct parse trees that reach them. We
// compute, for each (element, start-index), the set of reachable end-indices,
// deduplicating positions at every step and memoizing per (element, index).
// Positions live in [0, len(input)], so the work is polynomial rather than the
// exponential path enumeration that Parse performs.
//
// IsValid enforces left-termination before calling in, so the in-progress
// guard below is defense-in-depth: it can never fire for a left-terminating
// grammar (a same-index re-entry implies the rule is not left terminating).

type recognizer struct {
	g          *Grammar
	input      []byte
	memo       map[string]map[int]bool // element.String()+"@"+index -> reachable end set
	inProgress map[string]bool

	// Left-recursion support. leftRec maps each left-recursive rule (lowercase
	// name) to its left-corner SCC, grown jointly. growing holds the live seed
	// of each rule currently being grown (key "rule@index"); a re-entrant call
	// returns the seed instead of recursing.
	leftRec map[string][]string
	growing map[string]map[int]bool
}

func cloneSet(s map[int]bool) map[int]bool {
	out := make(map[int]bool, len(s))
	for k := range s {
		out[k] = true
	}
	return out
}

func (r *recognizer) reachAlt(alt Alternation, index int) map[int]bool {
	out := map[int]bool{}
	for _, c := range alt.Concatenations {
		for e := range r.reachConcat(c, index) {
			out[e] = true
		}
	}
	return out
}

func (r *recognizer) reachConcat(c Concatenation, index int) map[int]bool {
	cur := map[int]bool{index: true}
	for _, rep := range c.Repetitions {
		next := map[int]bool{}
		for p := range cur {
			for e := range r.reachRep(rep, p) {
				next[e] = true
			}
		}
		cur = next
		if len(cur) == 0 {
			break
		}
	}
	return cur
}

func (r *recognizer) reachRep(rep Repetition, index int) map[int]bool {
	ends := map[int]bool{}
	if rep.Min == 0 {
		ends[index] = true
	}

	// level = positions reachable after exactly c occurrences.
	level := map[int]bool{index: true}
	// seen = every position reached at any count so far. reachElem only moves
	// forward (an end position is >= its start), so once a step introduces no
	// position outside seen, the reachable set has closed and no later
	// repetition can add a new end position. Breaking there keeps an unbounded
	// repetition over an element with large reach-sets (e.g. *(1*ALPHA)) at
	// O(n^2); the previous len(next)==len(level) test never tripped because
	// level shrank by one each step ({1..n}, {2..n}, ...), giving O(n^3).
	seen := map[int]bool{index: true}

	// Levels are stationary after at most len(input)+1 counts: reachElem only
	// moves forward, so whether position x is in a level depends on positions
	// below x plus x reaching itself, and each position settles one count after
	// the ones below it. The stationary check below therefore fires within cap.
	cap := len(r.input) + 2
	for c := 1; c <= cap; c++ {
		if rep.Max != inf && c > rep.Max {
			break
		}
		next := map[int]bool{}
		for p := range level {
			for e := range r.reachElem(rep.Element, p) {
				next[e] = true
			}
		}
		if len(next) == 0 {
			break
		}
		if c >= rep.Min {
			for e := range next {
				ends[e] = true
			}
		}
		grew := false
		for e := range next {
			if !seen[e] {
				seen[e] = true
				grew = true
			}
		}
		// Stationary level: every further count reaches the same positions, so
		// a minimum not yet met is met by that same set. This happens with a
		// nullable element, e.g. 3"" on an empty input, where the count must
		// go past the input length to reach the minimum.
		if c < rep.Min && sameSet(next, level) {
			if rep.Max == inf || rep.Min <= rep.Max {
				for e := range next {
					ends[e] = true
				}
			}
			break
		}
		level = next
		// Reachable set closed and the minimum-count requirement met.
		if !grew && c >= rep.Min {
			break
		}
	}
	return ends
}

func (r *recognizer) reachElem(elem ElemItf, index int) map[int]bool {
	// Left-recursive rulenames are resolved by seed-growing rather than the
	// flat memoized recursion below (which would cut the recursion to empty and
	// under-accept).
	if rn, ok := elem.(ElemRulename); ok {
		if scc, isLR := r.leftRec[strings.ToLower(rn.Name)]; isLR {
			return r.reachLeftRec(strings.ToLower(rn.Name), scc, index)
		}
	}

	key := elem.String() + "@" + strconv.Itoa(index)
	if cached, ok := r.memo[key]; ok {
		return cached
	}
	if r.inProgress[key] {
		// Re-entry at the same position == non-progressing recursion in a rule
		// not flagged left-recursive. Return empty (defense-in-depth) instead of
		// looping forever.
		return map[int]bool{}
	}
	r.inProgress[key] = true
	out := r.computeElem(elem, index)
	delete(r.inProgress, key)
	// Safe to cache: a rule reached here either is independent of any actively
	// growing seed, or is itself a growing SCC member (handled by reachLeftRec
	// via r.growing, not this path). A lower left-corner SCC cannot reference
	// the SCC growing above it -- they would be one SCC -- so its result is
	// stable. Caching nested SCCs is what keeps left recursion polynomial.
	r.memo[key] = cloneSet(out)
	return out
}

// reachLeftRec computes the reachable end-set of a left-recursive rule at index
// by growing the whole left-corner SCC from the empty seed to its least
// fixpoint. Each growth step adds end-positions (bounded by len(input)), so it
// terminates; a re-entrant call to any SCC member at the same index returns the
// current seed. Pure (base-less) left recursion correctly settles at the empty
// set, i.e. "no match".
func (r *recognizer) reachLeftRec(name string, scc []string, index int) map[int]bool {
	sidx := "@" + strconv.Itoa(index)
	key := name + sidx
	if cached, ok := r.memo[key]; ok {
		return cached
	}
	if seed, ok := r.growing[key]; ok {
		return seed // re-entrant: return the current seed
	}

	keys := make([]string, len(scc))
	for i, x := range scc {
		keys[i] = x + sidx
		r.growing[keys[i]] = map[int]bool{}
	}
	for changed := true; changed; {
		changed = false
		for i, x := range scc {
			rule := GetRule(x, r.g.Rulemap)
			if rule == nil {
				continue
			}
			ns := r.reachAlt(rule.Alternation, index)
			cur := r.growing[keys[i]]
			for e := range ns {
				if !cur[e] {
					cur[e] = true
					changed = true
				}
			}
		}
	}

	// The fixpoint converged. These end-sets are final and independent of any
	// outer growing seed (the SCC only re-references its own members, held in
	// r.growing, and strictly lower SCCs, which cannot reference it back), so
	// they can be memoized unconditionally.
	var result map[int]bool
	for i, x := range scc {
		v := r.growing[keys[i]]
		delete(r.growing, keys[i])
		r.memo[keys[i]] = cloneSet(v)
		if x == name {
			result = cloneSet(v)
		}
	}
	return result
}

func (r *recognizer) computeElem(elem ElemItf, index int) map[int]bool {
	out := map[int]bool{}
	switch v := elem.(type) {
	case ElemRulename:
		rule := GetRule(v.Name, r.g.Rulemap)
		if rule == nil {
			return out
		}
		return r.reachAlt(rule.Alternation, index)

	case ElemGroup:
		return r.reachAlt(v.Alternation, index)

	case ElemOption:
		out[index] = true // 0 occurrences
		for e := range r.reachAlt(v.Alternation, index) {
			out[e] = true
		}
		return out

	case ElemProseVal:
		// prose-val can't be matched (same as the original parser)
		return out

	case ElemNumVal:
		switch v.Status {
		case StatRange:
			if index >= len(r.input) {
				return out
			}
			// Bounds are numeric: a range straddling the Unicode ceiling still
			// matches its representable (rune) subset, one wholly above it matches
			// nothing. numvalToRune never panics (saturates out-of-int32 bounds).
			min := numvalToRune(v.Elems[0], v.Base)
			max := numvalToRune(v.Elems[1], v.Base)
			ru, size := utf8.DecodeRune(r.input[index:])
			if ru == utf8.RuneError && size == 1 {
				return out
			}
			if min <= ru && ru <= max {
				out[index+size] = true
			}
		case StatSeries:
			idx := index
			for i := 0; i < len(v.Elems); i++ {
				ru := numvalToRune(v.Elems[i], v.Base)
				// A series element must denote a real character; an out-of-range
				// value can never appear in UTF-8 input, so the series matches
				// nothing (and we avoid a spurious U+FFFD coincidence).
				if !utf8.ValidRune(ru) {
					return out
				}
				s := string(ru)
				sz := len([]byte(s))
				if idx+sz > len(r.input) || s != string(r.input[idx:idx+sz]) {
					return out
				}
				idx += sz
			}
			out[idx] = true
		}
		return out

	case ElemCharVal:
		idx := index
		for i := 0; i < len(v.Values); i++ {
			if idx >= len(r.input) {
				return out
			}
			ru, size := utf8.DecodeRune(r.input[idx:])
			if ru == utf8.RuneError && size == 1 {
				return out
			}
			if !sensequal(v.Values[i], ru, v.Sensitive) {
				return out
			}
			idx += size
		}
		out[idx] = true
		return out
	}
	return out
}

func sameSet(a, b map[int]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}
