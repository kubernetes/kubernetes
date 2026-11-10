package goabnf

import (
	"fmt"
	"math/rand"
	"unicode/utf8"
)

// fuzz.go turns a (fully expanded) transition graph into a total
// function from arbitrary bytes to a string the grammar admits. It is the core
// primitive for structure-aware fuzzing with testing.F: because every input
// maps to a valid production, the fuzzing engine's mutations explore the
// grammar's structure instead of bouncing off the target's first syntax check.
//
// The walk is deliberately different from a TransitionGraphReader: the reader
// enumerates entrypoint->endpoint trails (each directed edge at most once) for
// finite coverage, whereas the generator takes a free walk that MAY repeat
// edges -- so it can unroll a "*" repetition into "a", "aa", "aaa", ... -- and
// is kept finite by explicit budgets (max length, per-edge repetitions, total
// steps) rather than by the no-repeat rule. When a budget is hit the walk does
// not stop dead (that could leave it on a non-accepting node); it steers along
// shortest paths to the nearest endpoint, so the output is always a complete,
// valid production.

// source supplies the decisions that drive a walk. Two implementations exist: a
// byte tape (for testing.F, where mutations of the tape are the fuzzing signal)
// and a *rand.Rand (for property tests and corpus generation). intn(n) returns
// a value in [0, n); when a tape is exhausted it returns 0, which makes
// generation total and biases an empty/short tape toward the shortest
// production.
type source interface {
	intn(n int) int
}

type tapeSource struct {
	tape []byte
	pos  int
}

func (s *tapeSource) intn(n int) int {
	if n <= 1 {
		return 0
	}
	// Read just enough bytes to address n, big-endian. One byte per decision
	// when n <= 256 keeps locality high: a single-byte mutation flips at most
	// one decision (until the number of decisions changes).
	width := 1
	for max := n; max > 256; max = (max + 255) / 256 {
		width++
	}
	v := 0
	for i := 0; i < width; i++ {
		b := 0
		if s.pos < len(s.tape) {
			b = int(s.tape[s.pos])
			s.pos++
		}
		v = v<<8 | b
		if v < 0 { // overflow guard on absurd widths
			v &= 0x7fffffff
		}
	}
	return v % n
}

type randSource struct{ r *rand.Rand }

func (s randSource) intn(n int) int {
	if n <= 1 {
		return 0
	}
	return s.r.Intn(n)
}

// indexedSource addresses the tape by decision number rather than a moving
// cursor: the i-th decision always reads the fixed-width slot [i*stride, ...],
// regardless of how many bytes earlier decisions "needed". This keeps a tape
// edit local -- it can only perturb the one decision whose slot it lands in --
// so an upstream structural change no longer reframes the bytes every later
// decision reads (the residual non-locality is purely semantic: a forked walk
// reaches different nodes, where the same slot byte means a different choice).
// stride 4 addresses any choice count up to 2^32, well above the widest
// (full-Unicode) num-val range.
type indexedSource struct {
	tape []byte
	i    int
}

const indexedStride = 4

func (s *indexedSource) intn(n int) int {
	if n <= 1 {
		return 0
	}
	off := s.i * indexedStride
	s.i++
	v := 0
	for k := range indexedStride {
		b := 0
		if off+k < len(s.tape) {
			b = int(s.tape[off+k])
		}
		v = v<<8 | b
		if v < 0 {
			v &= 0x7fffffff
		}
	}
	return v % n
}

// Generator produces strings admitted by a transition graph.
type Generator struct {
	tg *TransitionGraph

	maxLength int
	maxReps   int
	maxSteps  int

	stableAddr bool

	endSet    map[string]bool // node ID -> is an endpoint ("" == emptyNode)
	canReach  map[string]bool // node ID -> can still reach some endpoint
	distToEnd map[string]int  // node ID -> fewest steps to an endpoint
	entries   []*Node         // entrypoints that can reach an endpoint

	emitCache map[string][]emission // node ID -> octet emissions (lazy, for ExpectedNext)
	emitOK    map[string]bool       // node ID -> emissions analyzable
}

// GenOption configures a Generator.
type GenOption interface {
	apply(*Generator)
}

type genOptionFunc func(*Generator)

func (f genOptionFunc) apply(g *Generator) {
	f(g)
}

// WithMaxLength bounds the number of bytes a generated production may contain.
// Once reached the walk steers to the nearest endpoint. Defaults to 4096.
func WithMaxLength(n int) GenOption {
	return genOptionFunc(func(g *Generator) {
		g.maxLength = n
	})
}

// WithMaxReps bounds how many times a single edge may be traversed, which caps
// how far repetitions ("*" / "1*" / "n*m") are unrolled. Defaults to 32.
func WithMaxReps(n int) GenOption {
	return genOptionFunc(func(g *Generator) {
		g.maxReps = n
	})
}

// WithMaxSteps is a hard backstop on the total number of walk steps, defending
// against pathological graphs. Defaults to 100000.
func WithMaxSteps(n int) GenOption {
	return genOptionFunc(func(g *Generator) {
		g.maxSteps = n
	})
}

// WithStableAddressing makes Generate address the tape by decision number
// (fixed-width slots) instead of a moving cursor. This improves locality for a
// coverage-guided engine -- a one-byte tape edit perturbs a single decision
// rather than reframing every decision downstream of it -- at the cost of
// consuming the tape less densely. Recommended for use under testing.F.
func WithStableAddressing() GenOption {
	return genOptionFunc(func(g *Generator) {
		g.stableAddr = true
	})
}

// NewGenerator prepares a Generator from a transition graph. The graph must be
// fully expanded -- build it with WithDeflateRules(true) when the target rule
// references other rules -- because an unexpanded rule node carries no bytes and
// could not be turned into a valid production. It returns an error naming the
// first such node otherwise. (Recursive rules cannot be expanded at all; in that
// case TransitionGraph itself returns ErrCyclicRule before reaching here.)
func NewGenerator(tg *TransitionGraph, opts ...GenOption) (*Generator, error) {
	g := &Generator{
		tg:        tg,
		maxLength: 4096,
		maxReps:   32,
		maxSteps:  100_000,
	}
	for _, o := range opts {
		o.apply(g)
	}

	// Collect every reachable node and the reverse adjacency, and verify the
	// graph is fully expanded (only producing leaves and the empty node).
	revAdj := map[string][]*Node{}
	nodes := map[string]*Node{}
	var stack []*Node
	for _, e := range tg.Entrypoints {
		if e != emptyNode {
			stack = append(stack, e)
		}
	}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		id := tgNodeID(n)
		if _, ok := nodes[id]; ok {
			continue
		}
		nodes[id] = n
		if !producing(n) {
			return nil, fmt.Errorf("transition graph contains an unexpanded rule node %q; build it with WithDeflateRules(true)", n.Elem.String())
		}
		for _, sc := range n.Nexts {
			if sc == emptyNode {
				continue
			}
			revAdj[tgNodeID(sc)] = append(revAdj[tgNodeID(sc)], n)
			stack = append(stack, sc)
		}
	}

	// endSet, then backward BFS from endpoints for canReach + distToEnd.
	g.endSet = map[string]bool{}
	g.canReach = map[string]bool{}
	g.distToEnd = map[string]int{}
	var q []string
	for _, e := range tg.Endpoints {
		id := tgNodeID(e)
		g.endSet[id] = true
		if !g.canReach[id] {
			g.canReach[id] = true
			g.distToEnd[id] = 0
			q = append(q, id)
		}
	}
	for len(q) > 0 {
		id := q[0]
		q = q[1:]
		for _, p := range revAdj[id] {
			pid := tgNodeID(p)
			if g.canReach[pid] {
				continue
			}
			g.canReach[pid] = true
			g.distToEnd[pid] = g.distToEnd[id] + 1
			q = append(q, pid)
		}
	}

	for _, e := range tg.Entrypoints {
		if g.canReach[tgNodeID(e)] {
			g.entries = append(g.entries, e)
		}
	}
	return g, nil
}

// Generate maps the tape to a production the grammar admits. It is total: any
// tape (including nil) yields a valid production; an exhausted tape biases
// toward the shortest completion.
func (g *Generator) Generate(tape []byte) []byte {
	if g.stableAddr {
		return g.walk(&indexedSource{tape: tape}, nil)
	}
	return g.walk(&tapeSource{tape: tape}, nil)
}

// GenerateRand draws a production using r as the entropy source.
func (g *Generator) GenerateRand(r *rand.Rand) []byte {
	return g.walk(randSource{r: r}, nil)
}

// traceSeg records that out[off:off+length] was emitted by a node carrying elem.
type traceSeg struct {
	off, length int
	elem        ElemItf
}

func (g *Generator) walk(src source, trace *[]traceSeg) []byte {
	out := []byte{}
	if len(g.entries) == 0 {
		return out
	}
	node := g.entries[src.intn(len(g.entries))]

	visits := map[string]int{}
	steps := 0
	for {
		if node != emptyNode && producing(node) {
			_, vt := nodeEmit(node, 0)
			if vt < 1 {
				vt = 1
			}
			var idx int32
			if vt > 1 {
				idx = int32(src.intn(int(vt)))
			}
			b, _ := nodeEmit(node, idx)
			if trace != nil && len(b) > 0 {
				*trace = append(*trace, traceSeg{off: len(out), length: len(b), elem: node.Elem})
			}
			out = append(out, b...)
		}

		// The empty node is a terminal entry/endpoint sentinel with no
		// successors; reaching it as an entrypoint means the whole production is
		// nullable, so stopping here yields the (valid) empty production.
		if node == emptyNode {
			break
		}

		id := tgNodeID(node)
		atEnd := g.endSet[id]

		// liveAll: successors that can still reach an endpoint (used for
		// termination steering). liveCapped: those not yet at the per-edge cap
		// (used for ordinary random choice).
		var liveAll, liveCapped []*Node
		for _, sc := range node.Nexts {
			if sc == emptyNode || !g.canReach[tgNodeID(sc)] {
				continue
			}
			liveAll = append(liveAll, sc)
			if visits[id+">"+tgNodeID(sc)] < g.maxReps {
				liveCapped = append(liveCapped, sc)
			}
		}
		if len(liveAll) == 0 {
			break // terminal; canReach guarantees this node is an endpoint
		}

		over := len(out) >= g.maxLength || steps >= g.maxSteps
		terminating := over || len(liveCapped) == 0

		var next *Node
		if atEnd {
			if terminating {
				break
			}
			// One decision: 0 == stop here, otherwise continue to a successor.
			c := src.intn(len(liveCapped) + 1)
			if c == 0 {
				break
			}
			next = liveCapped[c-1]
		} else if terminating {
			next = g.nearestToEnd(liveAll)
		} else {
			next = liveCapped[src.intn(len(liveCapped))]
		}

		visits[id+">"+tgNodeID(next)]++
		steps++
		node = next
	}
	return out
}

// nearestToEnd returns the candidate with the fewest steps to an endpoint.
// Following it repeatedly walks a shortest path to an endpoint (distToEnd
// strictly decreases), so the termination phase never repeats an edge.
func (g *Generator) nearestToEnd(cands []*Node) *Node {
	best := cands[0]
	bestD := g.distToEnd[tgNodeID(best)]
	for _, c := range cands[1:] {
		if d := g.distToEnd[tgNodeID(c)]; d < bestD {
			best, bestD = c, d
		}
	}
	return best
}

// From here, generate *negative* test inputs: strings that lie just outside the
// grammar, each paired with a label describing the violation. They are the
// complement of Generate's valid productions -- a correct target MUST reject them
// -- and they reach the error-handling paths (truncation, boundary overruns,
// trailing input) where many parser bugs live and which a valid-only generator
// can never exercise.
//
// A perturbation is NOT assumed invalid: grammars can be ambiguous or redundant
// (flipping "a" to "b" in `"a" / "b"` is still valid), so every candidate is
// checked against the grammar and only genuinely-rejected ones are returned.
// This makes the "target must reject" oracle sound.

// NearMiss returns a string just outside the grammar together with a label
// describing how it violates the grammar, or ok=false if no minimal
// perturbation within the attempt budget produced a string the grammar rejects.
// It is deterministic in tape and total.
func (g *Generator) NearMiss(tape []byte) (production []byte, label string, ok bool) {
	src := &tapeSource{tape: tape}
	var trace []traceSeg
	base := g.walk(src, &trace)

	for _, c := range g.nearMissCandidates(src, base, trace) {
		if c.bytes == nil || string(c.bytes) == string(base) {
			continue
		}
		valid, err := g.tg.grammar.IsValid(g.tg.rulename, c.bytes)
		if err == nil && !valid {
			return c.bytes, c.label, true
		}
	}
	return nil, "", false
}

// NearMissRand is the *rand.Rand-driven variant of NearMiss.
func (g *Generator) NearMissRand(r *rand.Rand) (production []byte, label string, ok bool) {
	// Draw a tape and reuse the deterministic path so behaviour is identical.
	tape := make([]byte, 64)
	r.Read(tape)
	return g.NearMiss(tape)
}

type candidate struct {
	bytes []byte
	label string
}

func cloneBytes(b []byte) []byte {
	c := make([]byte, len(b))
	copy(c, b)
	return c
}

// nearMissCandidates proposes minimal perturbations of base, ordered with a
// little entropy from src so different tapes explore different operators first.
// Correctness does not depend on the order: NearMiss verifies each against the
// grammar.
func (g *Generator) nearMissCandidates(src source, base []byte, trace []traceSeg) []candidate {
	var cs []candidate

	// (1) Boundary violations: take a num-val range segment one step past its
	// edge. These are the high-value off-by-one cases.
	for _, seg := range trace {
		nv, isRange := numvalRange(seg.elem)
		if !isRange {
			continue
		}
		for _, b := range []struct {
			r    rune
			what string
		}{
			{rune(nv.min - 1), "below"},
			{rune(nv.max + 1), "above"},
		} {
			if b.r < 0 || b.r > 0x10FFFF {
				continue
			}
			cand := cloneBytes(base)
			repl := []byte(string(b.r))
			out := append(append(cloneBytes(cand[:seg.off]), repl...), cand[seg.off+seg.length:]...)
			cs = append(cs, candidate{out, fmt.Sprintf("byte at offset %d is one %s num-val range %s", seg.off, b.what, nv.label)})
		}
	}

	// (2) Off-alphabet substitution at a chosen position: replace one byte with
	// a "wild" value. Verified, so even an unlucky pick is safe.
	if len(base) > 0 {
		pos := src.intn(len(base))
		for _, nb := range []byte{0x00, 0xFF, base[pos] ^ 0x20, base[pos] + 1, '!'} {
			if nb == base[pos] {
				continue
			}
			cand := cloneBytes(base)
			cand[pos] = nb
			cs = append(cs, candidate{cand, fmt.Sprintf("byte at offset %d substituted with 0x%02x", pos, nb)})
		}
	}

	// (3) Truncation: drop a suffix (incomplete input -- exercises overread /
	// "premature end" handling).
	for _, n := range []int{1, 2, len(base) / 2, len(base) - 1} {
		if n <= 0 || n > len(base) {
			continue
		}
		cs = append(cs, candidate{cloneBytes(base[:len(base)-n]), fmt.Sprintf("truncated by %d byte(s) (len %d)", n, len(base)-n)})
	}

	// (4) Trailing input: append a byte (exercises "ignores trailing garbage").
	for _, nb := range []byte{0x00, '!', 0xFF, 'A'} {
		out := append(cloneBytes(base), nb)
		cs = append(cs, candidate{out, fmt.Sprintf("trailing byte 0x%02x appended", nb)})
	}

	// (5) Deletion of a single interior byte.
	if len(base) >= 2 {
		pos := src.intn(len(base))
		out := append(cloneBytes(base[:pos]), base[pos+1:]...)
		cs = append(cs, candidate{out, fmt.Sprintf("byte at offset %d deleted", pos)})
	}

	// Light entropy-driven rotation so the first verified hit varies by tape.
	if len(cs) > 1 {
		k := src.intn(len(cs))
		cs = append(cs[k:], cs[:k]...)
	}
	return cs
}

type numvalInfo struct {
	min, max int32
	label    string
}

// numvalRange reports whether elem is a num-val range and, if so, its bounds.
func numvalRange(elem ElemItf) (numvalInfo, bool) {
	nv, ok := elem.(ElemNumVal)
	if !ok || nv.Status != StatRange {
		return numvalInfo{}, false
	}
	min, max := numvalToInt32(nv.Elems[0], nv.Base), numvalToInt32(nv.Elems[1], nv.Base)
	// Clamp the upper bound to the Unicode range so the "just above" boundary
	// probe (max+1) stays a meaningful rune rather than overflowing.
	if max > utf8.MaxRune {
		max = utf8.MaxRune
	}
	return numvalInfo{min: min, max: max, label: elem.String()}, true
}

// From here answers for a given prefix "which octets may legally come next, and
// is the prefix itself a complete word?" by simulating the transition-graph
// automaton over the prefix and reading the frontier. The graph is not a plain
// byte DFA -- each node emits a multi-octet string with per-position choices
// (letter casings, num-val ranges) -- so this is an NFA-style frontier simulation,
// not a single-state walk.
//
// Its headline use is sharpening NearMiss: appending an octet that is NOT in the
// expected-next set yields a string that is provably not a prefix of any word in
// the language, i.e. guaranteed invalid, with a precise label. It also enables
// error-position testing ("the target must fail at byte N").

// octetSet is a 256-bit set of octets.
type octetSet [4]uint64

func (s *octetSet) add(b byte) {
	s[b>>6] |= 1 << (b & 63)
}

func (s octetSet) has(b byte) bool {
	return s[b>>6]&(1<<(b&63)) != 0
}

func (s octetSet) union(o octetSet) octetSet {
	return octetSet{s[0] | o[0], s[1] | o[1], s[2] | o[2], s[3] | o[3]}
}

func (s octetSet) bytes() []byte {
	var out []byte
	for b := range 256 {
		if s.has(byte(b)) {
			out = append(out, byte(b))
		}
	}
	return out
}

// emission is one fixed-length alternative of a node's output: pos[j] is the set
// of octets the node may emit at byte offset j. A node usually has exactly one
// emission; a num-val range that spans several UTF-8 lengths has one per length.
type emission struct {
	pos []octetSet
}

// expectedNextRangeCap bounds how many runes a multi-byte num-val range is
// expanded into for analysis; beyond it ExpectedNext reports ok=false rather
// than doing unbounded work. ASCII ranges are always handled (<=128 octets).
const expectedNextRangeCap = 1 << 16

func (g *Generator) emissions(n *Node) ([]emission, bool) {
	id := tgNodeID(n)
	if g.emitCache == nil {
		g.emitCache = map[string][]emission{}
		g.emitOK = map[string]bool{}
	}
	if e, seen := g.emitCache[id]; seen {
		return e, g.emitOK[id]
	}
	e, ok := nodeEmissions(n)
	g.emitCache[id] = e
	g.emitOK[id] = ok
	return e, ok
}

func nodeEmissions(n *Node) ([]emission, bool) {
	switch v := n.Elem.(type) {
	case ElemCharVal:
		e := emission{}
		for _, r := range v.Values {
			if !v.Sensitive && ((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')) {
				lo, up := r, r
				if lo >= 'A' && lo <= 'Z' {
					lo = lo - 'A' + 'a'
				}
				if up >= 'a' && up <= 'z' {
					up = up - 'a' + 'A'
				}
				var s octetSet
				s.add(byte(lo))
				s.add(byte(up))
				e.pos = append(e.pos, s)
			} else {
				for _, b := range []byte(string(r)) {
					var s octetSet
					s.add(b)
					e.pos = append(e.pos, s)
				}
			}
		}
		return []emission{e}, true

	case ElemNumVal:
		switch v.Status {
		case StatSeries:
			e := emission{}
			for _, el := range v.Elems {
				r := numvalToRune(el, v.Base)
				if !utf8.ValidRune(r) {
					// An out-of-range element can never appear in UTF-8 input.
					return nil, false
				}
				for _, b := range []byte(string(r)) {
					var s octetSet
					s.add(b)
					e.pos = append(e.pos, s)
				}
			}
			return []emission{e}, true
		case StatRange:
			min, max := numvalToInt32(v.Elems[0], v.Base), numvalToInt32(v.Elems[1], v.Base)
			// Clamp to the Unicode range first: otherwise an out-of-range bound
			// near MaxInt32 overflows the int32 count below, bypasses the cap
			// guard, and turns the enumeration loop into a DoS. A range wholly
			// above U+10FFFF emits nothing.
			if max > utf8.MaxRune {
				max = utf8.MaxRune
			}
			if min > utf8.MaxRune || min > max {
				return nil, false
			}
			if max <= 0x7f { // ASCII: a single one-octet emission
				var s octetSet
				for b := min; b <= max && b <= 0x7f; b++ { // explicit ASCII bound for the conversion
					s.add(byte(b))
				}
				return []emission{{pos: []octetSet{s}}}, true
			}
			if max-min+1 > expectedNextRangeCap {
				return nil, false
			}
			byLen := map[int]*emission{}
			for r := min; r <= max; r++ {
				bs := []byte(string(rune(r)))
				e := byLen[len(bs)]
				if e == nil {
					e = &emission{pos: make([]octetSet, len(bs))}
					byLen[len(bs)] = e
				}
				for j, b := range bs {
					e.pos[j].add(b)
				}
			}
			out := make([]emission, 0, len(byLen))
			for _, e := range byLen {
				out = append(out, *e)
			}
			return out, true
		}
	}
	return nil, false
}

// ExpectedNext simulates the automaton over prefix and reports the octets that
// may legally follow it, whether prefix is itself a complete word, and whether
// the analysis succeeded (ok is false only when the graph uses a feature this
// analyzer declines to expand, e.g. an enormous multi-byte num-val range). When
// prefix is not a viable prefix of any word, next is empty and accepts is false.
func (g *Generator) ExpectedNext(prefix []byte) (next []byte, accepts bool, ok bool) {
	type state struct {
		n    *Node
		e, p int
	}
	ready := []state{}
	seen := map[state]bool{}
	push := func(s state) {
		if !seen[s] {
			seen[s] = true
			ready = append(ready, s)
		}
	}
	start := func(n *Node) bool {
		ems, o := g.emissions(n)
		if !o {
			return false
		}
		for ei, em := range ems {
			if len(em.pos) > 0 {
				push(state{n, ei, 0})
			}
		}
		return true
	}
	for _, e := range g.tg.Entrypoints {
		if e == emptyNode {
			continue
		}
		if !start(e) {
			return nil, false, false
		}
	}

	lastCompletedEndpoint := false
	consumedAll := true
	for idx, b := range prefix {
		next2 := []state{}
		seen2 := map[state]bool{}
		push2 := func(s state) {
			if !seen2[s] {
				seen2[s] = true
				next2 = append(next2, s)
			}
		}
		lastCompletedEndpoint = false
		for _, s := range ready {
			ems, _ := g.emissions(s.n)
			em := ems[s.e]
			if !em.pos[s.p].has(b) {
				continue
			}
			if s.p+1 == len(em.pos) { // node s.n completes on this octet
				if g.endSet[tgNodeID(s.n)] {
					lastCompletedEndpoint = true
				}
				for _, m := range s.n.Nexts {
					if m == emptyNode {
						continue
					}
					ems2, o := g.emissions(m)
					if !o {
						return nil, false, false
					}
					for ei, em2 := range ems2 {
						if len(em2.pos) > 0 {
							push2(state{m, ei, 0})
						}
					}
				}
			} else {
				push2(state{s.n, s.e, s.p + 1})
			}
		}
		ready, seen = next2, seen2
		if len(ready) == 0 {
			if idx != len(prefix)-1 {
				consumedAll = false // dead before the end: prefix not viable
			}
			break
		}
	}

	if !consumedAll {
		return nil, false, true
	}
	if len(prefix) == 0 {
		accepts = g.endSet[""] // empty word accepted iff emptyNode is an endpoint
	} else {
		accepts = lastCompletedEndpoint
	}
	var acc octetSet
	for _, s := range ready {
		ems, _ := g.emissions(s.n)
		acc = acc.union(ems[s.e].pos[s.p])
	}
	return acc.bytes(), accepts, true
}
