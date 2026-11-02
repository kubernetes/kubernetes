package goabnf

import (
	"fmt"
	"math/rand"
	"unicode/utf8"
)

// Generates strings by recursive descent over the grammar AST, so unlike the*
// transition-graph Generator (a finite automaton, which can only express
// *regular* rules) it handles genuinely recursive rules -- nested objects,
// balanced delimiters, expression grammars -- where a rule is reached many times
// at many depths.
//
// The hard part of recursive generation is termination: a recursive rule can
// expand forever. This generator guarantees a finite, valid production by
// precomputing each rule's minimum expansion cost and, whenever a depth or
// length budget is exceeded, steering every choice toward the cheapest-to-finish
// expansion. It never truncates mid-derivation, so the output is always a
// complete word of the grammar. (It shares the tape/rand entropy `source` with
// the transition-graph generator, so it slots into testing.F the same way.)

const astUnbounded = 1 << 30 // "infinite" min-length sentinel (non-productive)

// ASTGenerator generates strings admitted by a (possibly recursive) grammar
// rule, directly from the AST.
type ASTGenerator struct {
	g     *Grammar
	start *Rule

	maxDepth  int // rule-expansion depth before steering to terminate
	maxLen    int // output bytes before steering to terminate
	maxRepeat int // cap for unbounded ("*") repetitions

	minCost map[string]int // rule name -> minimum expansion cost (productivity)
}

// ASTGenOption configures an ASTGenerator.
type ASTGenOption interface {
	apply(*ASTGenerator)
}

type astGenOptionFunc func(*ASTGenerator)

func (f astGenOptionFunc) apply(g *ASTGenerator) {
	f(g)
}

// WithMaxDepth bounds rule-expansion (recursion) depth; past it the generator
// steers toward terminating expansions. Defaults to 32.
func WithMaxDepth(n int) ASTGenOption {
	return astGenOptionFunc(func(g *ASTGenerator) {
		g.maxDepth = n
	})
}

// WithMaxLen bounds the output length; past it the generator steers toward
// terminating expansions. Defaults to 4096. (Mandatory repetitions are always
// emitted, so the bound is soft -- a single huge fixed repetition can exceed it.)
func WithMaxLen(n int) ASTGenOption {
	return astGenOptionFunc(func(g *ASTGenerator) {
		g.maxLen = n
	})
}

// WithMaxRepeat caps how far an unbounded ("*" / "1*") repetition is unrolled.
// Defaults to 16.
func WithMaxRepeat(n int) ASTGenOption {
	return astGenOptionFunc(func(g *ASTGenerator) {
		g.maxRepeat = n
	})
}

// NewASTGenerator prepares a generator for rule. It errors if the rule is
// undefined, or if any rule reachable from it is undefined or non-productive
// (cannot derive any finite string), since neither can be generated. Unlike the
// transition-graph generator, recursive rules are fully supported.
func NewASTGenerator(g *Grammar, rule string, opts ...ASTGenOption) (*ASTGenerator, error) {
	start := GetRule(rule, g.Rulemap)
	if start == nil {
		return nil, &ErrRuleNotFound{Rulename: rule}
	}
	ag := &ASTGenerator{
		g:         g,
		start:     start,
		maxDepth:  32,
		maxLen:    4096,
		maxRepeat: 16,
	}
	for _, o := range opts {
		o.apply(ag)
	}

	// Least-fixpoint of minimum expansion cost over every rule reachable in
	// principle -- user rules and the built-in core rules (ALPHA, DIGIT, ...),
	// which GetRule resolves as a fallback but which are absent from Rulemap.
	// Keyed by canonical rule name (GetRule is case-insensitive and honors user
	// overrides of core rules). A rule stuck at the sentinel is non-productive.
	effective := map[string]*Rule{}
	for n := range g.Rulemap {
		if r := GetRule(n, g.Rulemap); r != nil {
			effective[r.Name] = r
		}
	}
	for n := range coreRules {
		if r := GetRule(n, g.Rulemap); r != nil {
			effective[r.Name] = r
		}
	}
	ag.minCost = make(map[string]int, len(effective))
	for name := range effective {
		ag.minCost[name] = astUnbounded
	}
	for changed := true; changed; {
		changed = false
		for name, r := range effective {
			if c := ag.costAlt(r.Alternation); c < ag.minCost[name] {
				ag.minCost[name] = c
				changed = true
			}
		}
	}

	// Every rule reachable from start must be defined and productive.
	seen := map[string]bool{start.Name: true}
	var visitAlt func(Alternation) error
	visitElem := func(e ElemItf) error {
		switch x := e.(type) {
		case ElemRulename:
			r := GetRule(x.Name, g.Rulemap)
			if r == nil {
				return fmt.Errorf("rule %q (reachable from %q) is undefined and cannot be generated", x.Name, rule)
			}
			if seen[r.Name] {
				return nil
			}
			seen[r.Name] = true
			if ag.minCost[r.Name] >= astUnbounded {
				return fmt.Errorf("rule %q (reachable from %q) is non-productive (cannot derive any finite string)", r.Name, rule)
			}
			return visitAlt(r.Alternation)
		case ElemGroup:
			return visitAlt(x.Alternation)
		case ElemOption:
			return visitAlt(x.Alternation)
		}
		return nil
	}
	visitAlt = func(a Alternation) error {
		for _, c := range a.Concatenations {
			for _, rep := range c.Repetitions {
				if err := visitElem(rep.Element); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if ag.minCost[start.Name] >= astUnbounded {
		return nil, fmt.Errorf("rule %q is non-productive (cannot derive any finite string)", rule)
	}
	if err := visitAlt(start.Alternation); err != nil {
		return nil, err
	}
	return ag, nil
}

// Generate maps the tape to a production the grammar admits. Total and
// deterministic: any tape (including nil) yields a valid production; an
// exhausted tape biases toward the shortest one.
func (ag *ASTGenerator) Generate(tape []byte) []byte {
	return ag.walk(&tapeSource{tape: tape})
}

// GenerateRand draws a production using r as the entropy source.
func (ag *ASTGenerator) GenerateRand(r *rand.Rand) []byte {
	return ag.walk(randSource{r: r})
}

func (ag *ASTGenerator) walk(src source) []byte {
	out := []byte{}
	ag.genAlt(src, &out, ag.start.Alternation, 0)
	return out
}

// over reports whether we should stop taking expansions for their own sake and
// steer toward the nearest terminal string.
func (ag *ASTGenerator) over(out *[]byte, depth int) bool {
	return depth >= ag.maxDepth || len(*out) >= ag.maxLen
}

func (ag *ASTGenerator) genAlt(src source, out *[]byte, alt Alternation, depth int) {
	cs := alt.Concatenations
	if len(cs) == 0 {
		return
	}
	idx := 0
	if ag.over(out, depth) {
		idx = ag.cheapestConcat(cs) // steer: pick the cheapest-to-finish branch
	} else if len(cs) > 1 {
		idx = src.intn(len(cs))
	}
	ag.genConcat(src, out, cs[idx], depth)
}

func (ag *ASTGenerator) genConcat(src source, out *[]byte, concat Concatenation, depth int) {
	for _, rep := range concat.Repetitions {
		ag.genRep(src, out, rep, depth)
	}
}

func (ag *ASTGenerator) genRep(src source, out *[]byte, rep Repetition, depth int) {
	lo := rep.Min
	hi := rep.Max
	if hi == inf || hi < lo { // unbounded: cap the optional unrolling
		hi = lo + ag.maxRepeat
	}
	n := lo
	if !ag.over(out, depth) && hi > lo {
		n = lo + src.intn(hi-lo+1)
	}
	for i := 0; i < n; i++ {
		ag.genElem(src, out, rep.Element, depth)
		// Stop optional extras once over length, but always emit the mandatory lo.
		if i+1 >= lo && len(*out) >= ag.maxLen {
			break
		}
	}
}

func (ag *ASTGenerator) genElem(src source, out *[]byte, elem ElemItf, depth int) {
	switch e := elem.(type) {
	case ElemRulename:
		// Guaranteed defined+productive by NewASTGenerator.
		r := GetRule(e.Name, ag.g.Rulemap)
		ag.genAlt(src, out, r.Alternation, depth+1)

	case ElemGroup:
		ag.genAlt(src, out, e.Alternation, depth)

	case ElemOption:
		if !ag.over(out, depth) && src.intn(2) == 1 {
			ag.genAlt(src, out, e.Alternation, depth)
		}

	case ElemCharVal:
		for _, r := range e.Values {
			if !e.Sensitive && isASCIILetter(r) && src.intn(2) == 1 {
				r = flipASCIICase(r)
			}
			*out = append(*out, []byte(string(r))...)
		}

	case ElemNumVal:
		switch e.Status {
		case StatSeries:
			for _, v := range e.Elems {
				r := numvalToRune(v, e.Base)
				// Emit only real characters; an out-of-range value cannot appear
				// in UTF-8 input, so the series would never match anyway.
				if utf8.ValidRune(r) {
					*out = append(*out, []byte(string(r))...)
				}
			}
		case StatRange:
			min, max := numvalToRune(e.Elems[0], e.Base), numvalToRune(e.Elems[1], e.Base)
			// Emit within the representable (Unicode) subset of the range.
			if max > utf8.MaxRune {
				max = utf8.MaxRune
			}
			if min > utf8.MaxRune || min > max {
				break // nothing representable to emit
			}
			span := int(max-min) + 1
			if span < 1 {
				span = 1
			}
			*out = append(*out, []byte(string(min+rune(src.intn(span))))...)
		}

	case ElemProseVal:
		// prose-val is informal text; nothing mechanical to emit.
	}
}

// cheapestConcat returns the index of the concatenation with the smallest
// minimum expansion cost -- the fastest way to finish the current alternation.
func (ag *ASTGenerator) cheapestConcat(cs []Concatenation) int {
	best, bestCost := 0, ag.costConcat(cs[0])
	for i := 1; i < len(cs); i++ {
		if c := ag.costConcat(cs[i]); c < bestCost {
			best, bestCost = i, c
		}
	}
	return best
}

// ---- minimum-expansion-cost cost model (for productivity + steering) ----

func (ag *ASTGenerator) costAlt(alt Alternation) int {
	if len(alt.Concatenations) == 0 {
		return 0
	}
	best := astUnbounded
	for _, c := range alt.Concatenations {
		if v := ag.costConcat(c); v < best {
			best = v
		}
	}
	return best
}

func (ag *ASTGenerator) costConcat(concat Concatenation) int {
	total := 0
	for _, rep := range concat.Repetitions {
		total = addClamp(total, mulClamp(rep.Min, ag.costElem(rep.Element)))
	}
	return total
}

func (ag *ASTGenerator) costElem(elem ElemItf) int {
	switch e := elem.(type) {
	case ElemRulename:
		r := GetRule(e.Name, ag.g.Rulemap)
		if r == nil {
			return astUnbounded // undefined rule
		}
		if c, ok := ag.minCost[r.Name]; ok {
			// +1 expansion penalty: referencing a rule always costs strictly
			// more than emitting a terminal, so a recursive branch can never
			// tie with (and be chosen over) a terminating one -- even when a
			// nullable prefix makes their emitted lengths equal.
			return addClamp(1, c)
		}
		return astUnbounded
	case ElemGroup:
		return ag.costAlt(e.Alternation)
	case ElemOption:
		return 0 // can be skipped
	case ElemCharVal:
		return len(string(e.Values))
	case ElemNumVal:
		switch e.Status {
		case StatSeries:
			n := 0
			for _, v := range e.Elems {
				n += len(string(numvalToRune(v, e.Base)))
			}
			return n
		default: // StatRange: one rune, take the lower bound's encoded length
			return len(string(numvalToRune(e.Elems[0], e.Base)))
		}
	case ElemProseVal:
		return 0
	}
	return 0
}

func addClamp(a, b int) int {
	if a >= astUnbounded || b >= astUnbounded {
		return astUnbounded
	}
	if s := a + b; s < astUnbounded {
		return s
	}
	return astUnbounded
}

func mulClamp(n, c int) int {
	if n == 0 {
		return 0
	}
	if c >= astUnbounded {
		return astUnbounded
	}
	if p := n * c; p >= 0 && p < astUnbounded {
		return p
	}
	return astUnbounded
}

func isASCIILetter(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}

func flipASCIICase(r rune) rune {
	switch {
	case r >= 'a' && r <= 'z':
		return r - 'a' + 'A'
	case r >= 'A' && r <= 'Z':
		return r - 'A' + 'a'
	}
	return r
}
