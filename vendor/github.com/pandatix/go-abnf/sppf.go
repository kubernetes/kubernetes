package goabnf

import (
	"fmt"
	"math/big"
	"strconv"
	"unicode/utf8"
)

// This file implements a GLL parser (Scott & Johnstone) producing a binarized
// Shared Packed Parse Forest (SPPF) for ABNF grammars.
//
// GLL parses ANY context-free grammar -- including left, right, hidden and
// mutual recursion, and arbitrary ambiguity -- in worst-case O(n^3) time and
// space, using a graph-structured stack (GSS) and a descriptor worklist whose
// "seen" set makes recursion terminate. The forest shares every sub-derivation
// and packs ambiguity, so it stays polynomial even when it represents
// exponentially or infinitely many trees.
//
// Pipeline: lower the ABNF AST into a flat slotted grammar (synthetic
// nonterminals for groups, options and repetitions), then run GLL over the
// slots. Recognition, ambiguity, tree counting and extraction are polynomial
// queries over the resulting forest.

// ---------------------------------------------------------------------------
// Slotted grammar
// ---------------------------------------------------------------------------

type symKind uint8

const (
	symNonterm symKind = iota
	symTerm            // a char-val / num-val / prose-val matcher
	symEps             // matches the empty string
)

type ssym struct {
	kind symKind
	nt   int     // nonterminal id, for symNonterm
	term ElemItf // terminal element, for symTerm
}

type sgNT struct {
	label    string   // diagnostic label
	isRule   bool     // corresponds to an original ABNF rule (kept in the tree)
	ruleName string   // original rule name when isRule
	alts     [][]ssym // alternates; every alternate has >= 1 symbol

	// Native counted repetition. When isRep, the nonterminal stands for
	// `repElem{repMin,repMax}` and its two alternates are the self-recursive
	// encoding `R ::= eps | repElem R`. The [repMin,repMax] bound is enforced at
	// parse time by a count carried in the GLL state (see addRepAlts), so the
	// slot grammar stays O(1) in the repetition bound instead of unrolling it.
	isRep  bool
	repMin int
	repMax int // inf (== -1) means unbounded
}

type slotGrammar struct {
	nts   []*sgNT
	index map[string]int // dedup key -> nt id
	start int
	la    *lookahead // selection sets guarding descriptor creation
}

func (sg *slotGrammar) reserve(key, label string) (int, bool) {
	if id, ok := sg.index[key]; ok {
		return id, true
	}
	id := len(sg.nts)
	sg.nts = append(sg.nts, &sgNT{label: label})
	sg.index[key] = id
	return id, false
}

// compileSlots lowers grammar starting at rootRulename into a slotted grammar.
const defaultMaxSlots = 1 << 16

func compileSlots(g *Grammar, rootRulename string, maxSlots int) (*slotGrammar, error) {
	if GetRule(rootRulename, g.Rulemap) == nil {
		return nil, &ErrRuleNotFound{Rulename: rootRulename}
	}
	sg := &slotGrammar{index: map[string]int{}}
	c := &lowerer{g: g, sg: sg, max: maxSlots}
	sg.start = c.rule(rootRulename)
	if c.err != nil {
		return nil, c.err
	}
	if c.over {
		return nil, &ErrGrammarTooLarge{Max: maxSlots}
	}
	sg.la = computeLookahead(sg)
	return sg, nil
}

type lowerer struct {
	g    *Grammar
	sg   *slotGrammar
	err  error
	max  int  // max nonterminals; 0 == unbounded
	over bool // budget exceeded
}

func (c *lowerer) budgetExceeded() bool {
	if c.max > 0 && len(c.sg.nts) >= c.max {
		c.over = true
	}
	return c.over
}

func (c *lowerer) rule(name string) int {
	key := "rule:" + canon(name)
	id, existed := c.sg.reserve(key, name)
	if existed {
		return id
	}
	if c.budgetExceeded() {
		return id
	}
	rule := GetRule(name, c.g.Rulemap)
	if rule == nil {
		// Unknown rule: a nonterminal with no alternates derives nothing.
		c.sg.nts[id].label = name
		return id
	}
	c.sg.nts[id].isRule = true
	c.sg.nts[id].ruleName = rule.Name
	c.sg.nts[id].alts = c.alts(rule.Alternation)
	return id
}

func (c *lowerer) alts(a Alternation) [][]ssym {
	out := make([][]ssym, 0, len(a.Concatenations))
	for _, cc := range a.Concatenations {
		seq := make([]ssym, 0, len(cc.Repetitions))
		for _, rep := range cc.Repetitions {
			seq = append(seq, c.rep(rep))
		}
		if len(seq) == 0 {
			seq = []ssym{{kind: symEps}}
		}
		out = append(out, seq)
	}
	return out
}

func (c *lowerer) rep(rep Repetition) ssym {
	if rep.Min == 1 && rep.Max == 1 {
		return c.elem(rep.Element)
	}
	key := "rep:" + rep.String()
	id, existed := c.sg.reserve(key, rep.String())
	if existed {
		return ssym{kind: symNonterm, nt: id}
	}
	if c.budgetExceeded() {
		return ssym{kind: symNonterm, nt: id}
	}
	// A malformed bound where max < min accepts nothing: leave the rep
	// nonterminal with no alternates so it derives the empty language (it is
	// unsatisfiable, NOT unbounded). This matches the recognizer.
	if rep.Max != inf && rep.Max < rep.Min {
		return ssym{kind: symNonterm, nt: id}
	}
	// Lower to the self-recursive encoding R ::= eps | E R, a single nonterminal
	// regardless of the bound. The [Min,Max] count is enforced at parse time
	// (addRepAlts), so lowering is O(1) in the bound and there is no per-count
	// nonterminal explosion (the old optTail/reqOne unrolling, which scaled with
	// Max-Min and Min respectively).
	esym := c.elem(rep.Element)
	self := ssym{kind: symNonterm, nt: id}
	c.sg.nts[id].alts = [][]ssym{
		{{kind: symEps}}, // alt 0: stop
		{esym, self},     // alt 1: consume one element, then loop
	}
	c.sg.nts[id].label = rep.String()
	c.sg.nts[id].isRep = true
	c.sg.nts[id].repMin = rep.Min
	c.sg.nts[id].repMax = rep.Max
	return self
}

func (c *lowerer) elem(e ElemItf) ssym {
	switch v := e.(type) {
	case ElemRulename:
		return ssym{kind: symNonterm, nt: c.rule(v.Name)}
	case ElemGroup:
		return ssym{kind: symNonterm, nt: c.group(v)}
	case ElemOption:
		return ssym{kind: symNonterm, nt: c.option(v)}
	case ElemCharVal, ElemNumVal, ElemProseVal:
		return ssym{kind: symTerm, term: e}
	}
	return ssym{kind: symTerm, term: e}
}

func (c *lowerer) group(v ElemGroup) int {
	key := "grp:" + v.String()
	id, existed := c.sg.reserve(key, v.String())
	if existed {
		return id
	}
	if c.budgetExceeded() {
		return id
	}
	c.sg.nts[id].alts = c.alts(v.Alternation)
	return id
}

func (c *lowerer) option(v ElemOption) int {
	key := "opt:" + v.String()
	id, existed := c.sg.reserve(key, v.String())
	if existed {
		return id
	}
	if c.budgetExceeded() {
		return id
	}
	alts := [][]ssym{{{kind: symEps}}}
	alts = append(alts, c.alts(v.Alternation)...)
	c.sg.nts[id].alts = alts
	return id
}

func canon(name string) string {
	b := []byte(name)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 'a' - 'A'
		}
	}
	return string(b)
}

// ---------------------------------------------------------------------------
// SPPF nodes
// ---------------------------------------------------------------------------

type gnodeKind uint8

const (
	gSymbol gnodeKind = iota // a nonterminal node (Start..End)
	gInter                   // a binarisation intermediate node
	gTerm                    // a terminal / epsilon node
)

type gnode struct {
	kind       gnodeKind
	nt         int // symbol nonterminal id, for gSymbol
	Start, End int
	packs      []pack // >1 pack == ambiguity
}

// pack is a packed node. It is labelled by the grammar slot it completes and
// its pivot (Scott & Johnstone, "GLL parse-tree generation", Science of
// Computer Programming 78(10), 2013, §3.1); the pivot is the left extent of the
// last child. Two alternates deriving the same span through the same children
// are distinct derivations, hence distinct packed nodes.
type pack struct {
	L    slot
	kids []*gnode // 1 or 2 children
}

func (n *gnode) addPack(L slot, children []*gnode) {
	for _, p := range n.packs {
		if p.L != L || len(p.kids) != len(children) {
			continue
		}
		same := true
		for i := range p.kids {
			if p.kids[i] != children[i] {
				same = false
				break
			}
		}
		if same {
			return
		}
	}
	n.packs = append(n.packs, pack{L: L, kids: children})
}

// ---------------------------------------------------------------------------
// GLL machinery
// ---------------------------------------------------------------------------

type slot struct{ nt, alt, dot int }

type gssNode struct {
	ret     slot // return slot
	pos     int
	rc      int // repetition count carried for native counted repetition (0 otherwise)
	edges   []gssEdge
	edgeSet map[gssEdge]bool
}

type gssEdge struct {
	w  *gnode
	to *gssNode
}

type descriptor struct {
	L  slot
	u  *gssNode
	i  int
	w  *gnode
	rc int // repetition count (0 outside a counted repetition)
}

type gllParser struct {
	sg    *slotGrammar
	input []byte

	nodes  map[nodeKey]*gnode
	gss    map[gssKey]*gssNode
	u0     *gssNode
	seen   map[descriptor]bool
	work   []descriptor
	popped map[*gssNode]map[*gnode]bool

	maxNodes int
	aborted  bool

	// Furthest-failure diagnostics: maxPos is the deepest input offset at which
	// a terminal match was attempted and failed; expected collects the terminals
	// that could have been consumed there (deduplicated via expectedSeen). They
	// drive ParseError when the parse has no solution.
	maxPos       int
	expected     []string
	expectedSeen map[string]bool
}

// nodeKey interns SPPF nodes without allocating strings. Distinct kinds never
// collide: symbol nodes key on (kind,nt,start,end); intermediate nodes on
// (kind,L,start,end); terminal nodes on (kind,start,end).
type nodeKey struct {
	kind       gnodeKind
	nt         int
	L          slot
	start, end int
}

type gssKey struct {
	ret slot
	pos int
	rc  int
}

// addAlt adds the descriptor starting alternate L.alt of L.nt at i when the
// lookahead test admits it. A rejected alternate would fail on its first
// terminal at i (or, when nullable, its continuation would), so its terminals
// are reported to the furthest-failure diagnostics in its place.
func (p *gllParser) addAlt(L slot, u *gssNode, i int, rc int) {
	if p.admits(p.sg.la.alt(L.nt, L.alt), i) {
		p.add(L, u, i, nil, rc)
	}
}

// admits applies a lookahead test at i. A rejected sequence would have failed
// on one of the set's terminals at i, so those feed the furthest-failure
// diagnostics in its place.
func (p *gllParser) admits(s *laSet, i int) bool {
	if s.admits(p.input, i) {
		return true
	}
	if i >= p.maxPos {
		for _, t := range s.terms {
			p.noteFail(i, t)
		}
	}
	return false
}

func (p *gllParser) add(L slot, u *gssNode, i int, w *gnode, rc int) {
	d := descriptor{L, u, i, w, rc}
	if p.seen[d] {
		return
	}
	p.seen[d] = true
	p.work = append(p.work, d)
}

func (p *gllParser) gssNodeFor(ret slot, pos, rc int) *gssNode {
	k := gssKey{ret, pos, rc}
	if v, ok := p.gss[k]; ok {
		return v
	}
	v := &gssNode{ret: ret, pos: pos, rc: rc, edgeSet: map[gssEdge]bool{}}
	p.gss[k] = v
	return v
}

func (p *gllParser) create(L slot, u *gssNode, i int, w *gnode, rc int) *gssNode {
	v := p.gssNodeFor(L, i, rc)
	e := gssEdge{w: w, to: u}
	if !v.edgeSet[e] {
		v.edgeSet[e] = true
		v.edges = append(v.edges, e)
		for z := range p.popped[v] {
			y := p.getNodeP(L, w, z)
			p.add(L, u, z.End, y, rc)
		}
	}
	return v
}

func (p *gllParser) pop(u *gssNode, i int, z *gnode) {
	if u == p.u0 {
		return
	}
	if p.popped[u] == nil {
		p.popped[u] = map[*gnode]bool{}
	}
	p.popped[u][z] = true
	for _, e := range u.edges {
		y := p.getNodeP(u.ret, e.w, z)
		p.add(u.ret, e.to, i, y, u.rc)
	}
}

func (p *gllParser) findNode(k nodeKey) *gnode {
	if v, ok := p.nodes[k]; ok {
		return v
	}
	if p.maxNodes > 0 && len(p.nodes) >= p.maxNodes {
		p.aborted = true
	}
	v := &gnode{kind: k.kind, nt: k.nt, Start: k.start, End: k.end}
	p.nodes[k] = v
	return v
}

func (p *gllParser) getNodeT(start, end int) *gnode {
	return p.findNode(nodeKey{kind: gTerm, nt: -1, start: start, end: end})
}

// getNodeP combines left node w (may be nil == $) and right node z under slot L.
func (p *gllParser) getNodeP(L slot, w, z *gnode) *gnode {
	prod := p.sg.nts[L.nt].alts[L.alt]
	betaEmpty := L.dot == len(prod)
	if L.dot == 1 && !betaEmpty {
		return z // single symbol consumed, more to come: pass straight up
	}
	start := z.Start
	if w != nil {
		start = w.Start
	}
	var k nodeKey
	if betaEmpty {
		k = nodeKey{kind: gSymbol, nt: L.nt, start: start, end: z.End}
	} else {
		k = nodeKey{kind: gInter, nt: -1, L: L, start: start, end: z.End}
	}
	y := p.findNode(k)
	if w != nil {
		y.addPack(L, []*gnode{w, z})
	} else {
		y.addPack(L, []*gnode{z})
	}
	return y
}

// matchTerm returns the end index after matching the terminal at i, or -1.
func (p *gllParser) matchTerm(s ssym, i int) int {
	if s.kind == symEps {
		return i
	}
	switch v := s.term.(type) {
	case ElemCharVal:
		idx := i
		for k := 0; k < len(v.Values); k++ {
			if idx >= len(p.input) {
				return -1
			}
			r, size := utf8.DecodeRune(p.input[idx:])
			if r == utf8.RuneError && size == 1 {
				return -1
			}
			if !sensequal(v.Values[k], r, v.Sensitive) {
				return -1
			}
			idx += size
		}
		return idx
	case ElemNumVal:
		switch v.Status {
		case StatRange:
			if i >= len(p.input) {
				return -1
			}
			// Numeric comparison: a range straddling U+10FFFF matches its rune
			// subset; numvalToRune never panics (saturates out-of-int32 bounds).
			min := numvalToRune(v.Elems[0], v.Base)
			max := numvalToRune(v.Elems[1], v.Base)
			r, size := utf8.DecodeRune(p.input[i:])
			if r == utf8.RuneError && size == 1 {
				return -1
			}
			if min <= r && r <= max {
				return i + size
			}
			return -1
		case StatSeries:
			idx := i
			for k := 0; k < len(v.Elems); k++ {
				ru := numvalToRune(v.Elems[k], v.Base)
				// Series elements must be real characters: an out-of-range value
				// matches no UTF-8 input (and avoids a spurious U+FFFD match).
				if !utf8.ValidRune(ru) {
					return -1
				}
				str := string(ru)
				sz := len(str)
				if idx+sz > len(p.input) || str != string(p.input[idx:idx+sz]) {
					return -1
				}
				idx += sz
			}
			return idx
		}
	}
	return -1 // prose-val and anything else never matches
}

// parse runs the GLL loop and returns the symbol node (start, 0, n) or nil.
func (p *gllParser) parse() *gnode {
	startNT := p.sg.start
	p.u0 = &gssNode{ret: slot{-1, -1, -1}, pos: 0}
	for ai := range p.sg.nts[startNT].alts {
		p.addAlt(slot{startNT, ai, 0}, p.u0, 0, 0)
	}
	for len(p.work) > 0 {
		if p.aborted {
			return nil
		}
		d := p.work[len(p.work)-1]
		p.work = p.work[:len(p.work)-1]
		p.process(d)
	}
	key := nodeKey{kind: gSymbol, nt: startNT, start: 0, end: len(p.input)}
	return p.nodes[key]
}

func (p *gllParser) process(d descriptor) {
	L, u, i, w, rc := d.L, d.u, d.i, d.w, d.rc
	for {
		prod := p.sg.nts[L.nt].alts[L.alt]
		if L.dot == len(prod) {
			z := w
			if z == nil {
				z = p.getNodeT(i, i)
			}
			p.pop(u, i, z)
			return
		}
		s := prod[L.dot]
		if s.kind == symNonterm {
			if !p.admits(p.sg.la.callAt(L), i) {
				return
			}
			ret := slot{L.nt, L.alt, L.dot + 1}
			// Preserve the current repetition frame count across the sub-parse so
			// that when it returns we resume the loop with the right count.
			v := p.create(ret, u, i, w, rc)
			if p.sg.nts[s.nt].isRep {
				// Native counted repetition: a self-reference inside the rep's own
				// loop alternate continues the chain (count+1); any other reference
				// starts a fresh chain at count 0.
				childRC := 0
				if s.nt == L.nt {
					childRC = capRC(rc, p.sg.nts[s.nt])
				}
				p.addRepAlts(s.nt, v, i, childRC)
			} else {
				for ai := range p.sg.nts[s.nt].alts {
					p.addAlt(slot{s.nt, ai, 0}, v, i, 0)
				}
			}
			return
		}
		j := p.matchTerm(s, i)
		if j < 0 {
			p.noteFail(i, s)
			return
		}
		cR := p.getNodeT(i, j)
		next := slot{L.nt, L.alt, L.dot + 1}
		w = p.getNodeP(next, w, cR)
		i = j
		L = next
	}
}

// noteFail records that a terminal match failed at input offset i. It maintains
// the furthest such offset and, for that offset only, the set of terminals that
// could have been consumed -- the classic "furthest-failure" position used to
// report where parsing got stuck.
func (p *gllParser) noteFail(i int, s ssym) {
	if i > p.maxPos {
		p.maxPos = i
		p.expected = p.expected[:0]
		p.expectedSeen = nil
	}
	if i != p.maxPos {
		return
	}
	desc := "?"
	if str, ok := s.term.(fmt.Stringer); ok {
		desc = str.String()
	}
	if p.expectedSeen == nil {
		p.expectedSeen = map[string]bool{}
	}
	if !p.expectedSeen[desc] {
		p.expectedSeen[desc] = true
		p.expected = append(p.expected, desc)
	}
}

// capRC returns the count after consuming one more element of a repetition R
// currently at count rc. For an unbounded repetition it saturates at repMin
// (once the minimum is met the exact count no longer matters), which keeps the
// reachable counts -- and hence the GLL state -- finite even when the element is
// nullable.
func capRC(rc int, nt *sgNT) int {
	next := rc + 1
	if nt.repMax == inf && next > nt.repMin {
		next = nt.repMin
	}
	return next
}

// addRepAlts seeds the two alternates of a repetition nonterminal, gated by the
// current count rc: it may stop (alt 0) only once the minimum is met, and may
// consume another element (alt 1) only while under the maximum. This enforces
// the {repMin,repMax} bound in parser state instead of unrolling it into the
// grammar, mirroring the recognizer's c>=min / c<max gates. rc is carried in the
// GSS key so two derivations that reach the same point with different committed
// counts are kept distinct (no over-/under-counting via shared continuations).
func (p *gllParser) addRepAlts(ntID int, v *gssNode, i, rc int) {
	nt := p.sg.nts[ntID]
	if rc >= nt.repMin {
		p.addAlt(slot{ntID, 0, 0}, v, i, rc) // stop: alt 0 == [eps]
	}
	if nt.repMax == inf || rc < nt.repMax {
		p.addAlt(slot{ntID, 1, 0}, v, i, rc) // loop: alt 1 == [element, self]
	}
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Forest is a shared packed parse forest produced by ParseForest.
type Forest struct {
	sg       *slotGrammar
	input    []byte
	rulename string
	root     *gnode
	built    int

	// maxPos / expected carry the furthest-failure diagnostics from the parse,
	// surfaced through ParseError when the forest is invalid.
	maxPos   int
	expected []string
}

// ForestOption configures ParseForest.
type ForestOption func(*forestConfig)

type forestConfig struct {
	maxNodes int
	maxSlots int
}

// WithMaxForestNodes bounds the number of SPPF nodes the forest may allocate.
// Because a complete SPPF is worst-case cubic in the input length, this caps the
// time and memory ParseForest can spend on adversarial input. 0 means unbounded.
func WithMaxForestNodes(n int) ForestOption { return func(c *forestConfig) { c.maxNodes = n } }

// WithMaxSlots bounds the number of nonterminals produced while lowering the
// grammar. A finite repetition m*n lowers to ~n nonterminals, so this caps the
// cost of an adversarial grammar such as a = 9999999999"x". 0 means unbounded;
// the default is 65536.
func WithMaxSlots(n int) ForestOption { return func(c *forestConfig) { c.maxSlots = n } }

// ErrForestTooLarge is returned by ParseForest when the SPPF exceeds the
// configured WithMaxForestNodes budget.
type ErrForestTooLarge struct{ Max int }

func (e *ErrForestTooLarge) Error() string {
	return "abnf: parse forest exceeded " + strconv.Itoa(e.Max) + " nodes"
}

// ErrGrammarTooLarge is returned by ParseForest when lowering the grammar would
// exceed the WithMaxSlots budget, typically because of an absurd repetition
// bound (e.g. a = 9999999999"x").
type ErrGrammarTooLarge struct{ Max int }

func (e *ErrGrammarTooLarge) Error() string {
	return "abnf: grammar lowering exceeded " + strconv.Itoa(e.Max) + " nonterminals"
}

func newGLLParser(sg *slotGrammar, input []byte, maxNodes int) *gllParser {
	return &gllParser{
		sg:       sg,
		input:    input,
		nodes:    map[nodeKey]*gnode{},
		gss:      map[gssKey]*gssNode{},
		seen:     map[descriptor]bool{},
		popped:   map[*gssNode]map[*gnode]bool{},
		maxNodes: maxNodes,
	}
}

// ParseForest parses input with grammar starting at rootRulename using GLL,
// returning the shared packed parse forest. It runs in worst-case O(n^3) time
// and space for ANY grammar -- left/right/mutual recursion and ambiguity
// included -- so it is safe on untrusted or malformed grammars.
func ParseForest(input []byte, grammar *Grammar, rootRulename string, opts ...ForestOption) (*Forest, error) {
	cfg := forestConfig{maxSlots: defaultMaxSlots}
	for _, o := range opts {
		o(&cfg)
	}
	sg, err := compileSlots(grammar, rootRulename, cfg.maxSlots)
	if err != nil {
		return nil, err
	}
	p := newGLLParser(sg, input, cfg.maxNodes)
	root := p.parse()
	if p.aborted {
		return nil, &ErrForestTooLarge{Max: cfg.maxNodes}
	}
	return &Forest{sg: sg, input: input, rulename: rootRulename, root: root, built: len(p.nodes), maxPos: p.maxPos, expected: p.expected}, nil
}

// Valid reports whether the whole input is derivable by the root rule.
func (f *Forest) Valid() bool { return f.root != nil }

// ParseError returns the furthest-failure diagnostic for an invalid forest:
// where parsing got stuck and which terminals were expected there. It returns
// nil when the forest is valid. The returned error unwraps to
// [ErrNoSolutionFound].
func (f *Forest) ParseError() *ParseError {
	if f.Valid() {
		return nil
	}
	return newParseError(f.input, f.maxPos, f.expected)
}

// Nodes returns the number of forest nodes reachable from the root.
func (f *Forest) Nodes() int {
	if f.root == nil {
		return 0
	}
	seen := map[*gnode]bool{}
	var walk func(n *gnode)
	walk = func(n *gnode) {
		if seen[n] {
			return
		}
		seen[n] = true
		for _, pk := range n.packs {
			for _, c := range pk.kids {
				walk(c)
			}
		}
	}
	walk(f.root)
	return len(seen)
}

// Ambiguous reports whether the input has more than one distinct parse tree.
func (f *Forest) Ambiguous() bool {
	if f.root == nil {
		return false
	}
	seen := map[*gnode]bool{}
	var walk func(n *gnode) bool
	walk = func(n *gnode) bool {
		if seen[n] {
			return false
		}
		seen[n] = true
		if len(n.packs) > 1 {
			return true
		}
		for _, pk := range n.packs {
			for _, c := range pk.kids {
				if walk(c) {
					return true
				}
			}
		}
		return false
	}
	return walk(f.root)
}

// NumTrees returns the number of distinct parse trees the forest represents.
// A result of -1 means infinitely many (a cycle in the forest, i.e. an
// infinitely-ambiguous grammar such as A = A / "x").
func (f *Forest) NumTrees() *big.Int {
	if f.root == nil {
		return big.NewInt(0)
	}
	memo := map[*gnode]*big.Int{}
	onStack := map[*gnode]bool{}
	infinite := false
	var count func(n *gnode) *big.Int
	count = func(n *gnode) *big.Int {
		if v, ok := memo[n]; ok {
			return v
		}
		if onStack[n] {
			infinite = true
			return big.NewInt(1)
		}
		onStack[n] = true
		total := big.NewInt(0)
		if len(n.packs) == 0 {
			total = big.NewInt(1)
		}
		for _, pk := range n.packs {
			prod := big.NewInt(1)
			for _, c := range pk.kids {
				prod.Mul(prod, count(c))
			}
			total.Add(total, prod)
		}
		onStack[n] = false
		memo[n] = total
		return total
	}
	res := count(f.root)
	if infinite {
		return big.NewInt(-1)
	}
	return res
}

// ParseTree is a concrete syntax tree keyed by grammar rules: rule nodes carry
// the rule name, terminals are leaves with empty Rule.
type ParseTree struct {
	Rule       string
	Start, End int
	Children   []*ParseTree
}

// Tree extracts a single parse tree (first packing at each node), or nil if the
// input is invalid. A visited guard keeps extraction finite on cyclic forests.
func (f *Forest) Tree() *ParseTree {
	if f.root == nil {
		return nil
	}
	return f.emitSymbol(f.root, map[*gnode]bool{})
}

func (f *Forest) emitSymbol(n *gnode, onStack map[*gnode]bool) *ParseTree {
	t := &ParseTree{Start: n.Start, End: n.End}
	if n.kind == gSymbol && n.nt >= 0 && f.sg.nts[n.nt].isRule {
		t.Rule = f.sg.nts[n.nt].ruleName
	}
	onStack[n] = true
	f.collect(n, &t.Children, onStack)
	delete(onStack, n)
	return t
}

func (f *Forest) collect(n *gnode, into *[]*ParseTree, onStack map[*gnode]bool) {
	if len(n.packs) == 0 {
		if n.kind == gTerm && n.End > n.Start {
			*into = append(*into, &ParseTree{Start: n.Start, End: n.End})
		}
		return
	}
	for _, c := range n.packs[0].kids {
		f.collectChild(c, into, onStack)
	}
}

func (f *Forest) collectChild(c *gnode, into *[]*ParseTree, onStack map[*gnode]bool) {
	switch {
	case c.kind == gTerm:
		if c.End > c.Start {
			*into = append(*into, &ParseTree{Start: c.Start, End: c.End})
		}
	case c.kind == gSymbol && c.nt >= 0 && f.sg.nts[c.nt].isRule:
		if onStack[c] {
			*into = append(*into, &ParseTree{Rule: f.sg.nts[c.nt].ruleName, Start: c.Start, End: c.End})
			return
		}
		*into = append(*into, f.emitSymbol(c, onStack))
	default:
		if onStack[c] {
			return
		}
		onStack[c] = true
		f.collect(c, into, onStack)
		delete(onStack, c)
	}
}
