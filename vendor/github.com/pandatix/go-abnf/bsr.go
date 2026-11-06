package goabnf

import (
	"math/big"
	"unicode/utf8"
)

// Binary Subtree Representation (BSR).
//
// This is an alternative derivation representation to the SPPF in sppf.go,
// following Scott, Johnstone & van Binsbergen, "Derivation representation using
// binary subtree sets" (Science of Computer Programming 175, 2019). Instead of
// interning symbol/intermediate/packed nodes, the parser records a flat set of
// elements (slot, l, k, r): a grammar slot whose consumed prefix derives the
// input span [l,r], with the most recently consumed symbol spanning [k,r] and
// the earlier prefix spanning [l,k]. The same GLL control (descriptors, GSS,
// native counted repetition) drives it; only what gets recorded differs.
//
// The GSS node's pos field already holds a production's left extent (the call
// position), so a BSR element needs no extra threading: on a dot advance the
// element is (newSlot, l=frameLeftExtent, k=pivot, r=end).
//
// The BSR engine is validated to produce identical Valid/NumTrees/Ambiguous
// results to the SPPF engine across the differential corpus; it is the intended
// foundation for grammar-directed parser generation.

// bsrElem is one binary subtree element. The slot's dot is "just past" the most
// recently consumed symbol; that symbol spans [k,r] and the earlier prefix spans
// [l,k]. A complete production (dot == len) records the nonterminal's extent.
type bsrElem struct {
	slot    slot
	l, k, r int
}

// bsrGNode is a GSS node for the BSR engine. Unlike the SPPF GSS, its edges are
// unlabelled (no SPPF node is carried): the BSR pivots are recovered from node
// positions instead.
type bsrGNode struct {
	ret     slot // return slot in the caller's production
	pos     int  // call position == the caller production's left extent for callees
	rc      int  // repetition count (native counted repetition; 0 otherwise)
	edges   []*bsrGNode
	edgeSet map[*bsrGNode]bool
}

type bsrDesc struct {
	L  slot
	u  *bsrGNode
	i  int
	rc int
}

type bsrParser struct {
	sg    *slotGrammar
	input []byte

	set    map[bsrElem]bool
	gss    map[gssKey]*bsrGNode
	u0     *bsrGNode
	seen   map[bsrDesc]bool
	work   []bsrDesc
	popped map[*bsrGNode]map[int]bool

	maxElems int
	aborted  bool
}

// addAlt adds the descriptor starting alternate L.alt of L.nt at i when the
// lookahead test admits it.
func (p *bsrParser) addAlt(L slot, u *bsrGNode, i, rc int) {
	if p.sg.la.alt(L.nt, L.alt).admits(p.input, i) {
		p.add(L, u, i, rc)
	}
}

func (p *bsrParser) add(L slot, u *bsrGNode, i, rc int) {
	d := bsrDesc{L, u, i, rc}
	if p.seen[d] {
		return
	}
	p.seen[d] = true
	p.work = append(p.work, d)
}

func (p *bsrParser) gssNodeFor(ret slot, pos, rc int) *bsrGNode {
	k := gssKey{ret, pos, rc}
	if v, ok := p.gss[k]; ok {
		return v
	}
	v := &bsrGNode{ret: ret, pos: pos, rc: rc, edgeSet: map[*bsrGNode]bool{}}
	p.gss[k] = v
	return v
}

// record adds the BSR element for advancing into slot L over [l,r] with the last
// symbol pivoting at k. Per the binary-subtree scheme an element is kept only
// when the consumed prefix has length >= 2 (an intermediate extent) or the
// production is complete (dot == len). A length-1 prefix is degenerate (l == k)
// and is reconstructed directly from the single symbol, so it is not stored.
func (p *bsrParser) record(L slot, l, k, r int) {
	prod := p.sg.nts[L.nt].alts[L.alt]
	if L.dot >= 2 || L.dot == len(prod) {
		if p.maxElems > 0 && len(p.set) >= p.maxElems {
			p.aborted = true
			return
		}
		p.set[bsrElem{L, l, k, r}] = true
	}
}

func (p *bsrParser) create(ret slot, u *bsrGNode, i, rc int) *bsrGNode {
	v := p.gssNodeFor(ret, i, rc)
	if !v.edgeSet[u] {
		v.edgeSet[u] = true
		v.edges = append(v.edges, u)
		for j := range p.popped[v] {
			p.record(ret, u.pos, i, j)
			p.add(ret, u, j, rc)
		}
	}
	return v
}

func (p *bsrParser) pop(u *bsrGNode, j int) {
	if u == p.u0 {
		return
	}
	if p.popped[u] == nil {
		p.popped[u] = map[int]bool{}
	}
	p.popped[u][j] = true
	for _, v := range u.edges {
		p.record(u.ret, v.pos, u.pos, j)
		p.add(u.ret, v, j, u.rc)
	}
}

func (p *bsrParser) addRepAlts(ntID int, v *bsrGNode, i, rc int) {
	nt := p.sg.nts[ntID]
	if rc >= nt.repMin {
		p.addAlt(slot{ntID, 0, 0}, v, i, rc)
	}
	if nt.repMax == inf || rc < nt.repMax {
		p.addAlt(slot{ntID, 1, 0}, v, i, rc)
	}
}

func (p *bsrParser) matchTerm(s ssym, i int) int {
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
	return -1
}

func (p *bsrParser) parse() {
	startNT := p.sg.start
	p.u0 = &bsrGNode{ret: slot{-1, -1, -1}, pos: 0, edgeSet: map[*bsrGNode]bool{}}
	for ai := range p.sg.nts[startNT].alts {
		p.addAlt(slot{startNT, ai, 0}, p.u0, 0, 0)
	}
	for len(p.work) > 0 {
		if p.aborted {
			return
		}
		d := p.work[len(p.work)-1]
		p.work = p.work[:len(p.work)-1]
		p.process(d)
	}
}

func (p *bsrParser) process(d bsrDesc) {
	L, u, i, rc := d.L, d.u, d.i, d.rc
	for {
		prod := p.sg.nts[L.nt].alts[L.alt]
		if L.dot == len(prod) {
			p.pop(u, i)
			return
		}
		s := prod[L.dot]
		if s.kind == symNonterm {
			if !p.sg.la.callAt(L).admits(p.input, i) {
				return
			}
			ret := slot{L.nt, L.alt, L.dot + 1}
			v := p.create(ret, u, i, rc)
			if p.sg.nts[s.nt].isRep {
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
			return
		}
		next := slot{L.nt, L.alt, L.dot + 1}
		p.record(next, u.pos, i, j)
		i = j
		L = next
	}
}

// ---------------------------------------------------------------------------
// Queries over the BSR set
// ---------------------------------------------------------------------------

// BSRForest is the BSR analogue of Forest: a set of binary subtree elements plus
// an index for reconstruction. Valid/NumTrees/Ambiguous/Tree mirror Forest.
type BSRForest struct {
	sg       *slotGrammar
	input    []byte
	rulename string
	set      map[bsrElem]bool
	index    map[bsrKey][]int // (slot,l,r) -> pivots k
	start    int
	n        int
}

type bsrKey struct {
	slot slot
	l, r int
}

func (f *BSRForest) buildIndex() {
	f.index = make(map[bsrKey][]int, len(f.set))
	for e := range f.set {
		k := bsrKey{e.slot, e.l, e.r}
		f.index[k] = append(f.index[k], e.k)
	}
}

// startElems returns, for each production of the start rule, the pivots of the
// complete-production elements spanning the whole input.
func (f *BSRForest) startElems() (slot, []int, bool) {
	for ai, prod := range f.sg.nts[f.start].alts {
		s := slot{f.start, ai, len(prod)}
		if ks := f.index[bsrKey{s, 0, f.n}]; len(ks) > 0 {
			return s, ks, true
		}
	}
	return slot{}, nil, false
}

// Valid reports whether the whole input is derivable by the root rule.
func (f *BSRForest) Valid() bool {
	for ai, prod := range f.sg.nts[f.start].alts {
		s := slot{f.start, ai, len(prod)}
		if len(f.index[bsrKey{s, 0, f.n}]) > 0 {
			return true
		}
	}
	return false
}

// Nodes returns the number of BSR elements (the representation's size).
func (f *BSRForest) Nodes() int { return len(f.set) }

// symNode/interNode identities for memoisation and cycle detection.
type bsrNodeID struct {
	inter bool
	slot  slot // for intermediate nodes
	nt    int  // for symbol nodes
	l, r  int
}

// NumTrees returns the number of distinct parse trees represented, or -1 for
// infinitely many (a cycle, i.e. an infinitely-ambiguous grammar). It mirrors
// Forest.NumTrees but counts over BSR extents instead of SPPF nodes.
func (f *BSRForest) NumTrees() *big.Int {
	if !f.Valid() {
		return big.NewInt(0)
	}
	memo := map[bsrNodeID]*big.Int{}
	onStack := map[bsrNodeID]bool{}
	infinite := false

	var symWays func(s ssym, a, b int) *big.Int
	var countSym func(nt, l, r int) *big.Int
	var countInter func(sl slot, l, r int) *big.Int
	var countElem func(sl slot, l, k, r int) *big.Int

	symWays = func(s ssym, a, b int) *big.Int {
		if s.kind == symNonterm {
			return countSym(s.nt, a, b)
		}
		return big.NewInt(1) // terminal / eps leaf
	}

	countElem = func(sl slot, l, k, r int) *big.Int {
		prod := f.sg.nts[sl.nt].alts[sl.alt]
		last := symWays(prod[sl.dot-1], k, r)
		var pre *big.Int
		switch sl.dot {
		case 1: // empty prefix (l == k)
			pre = big.NewInt(1)
		case 2: // single-symbol prefix
			pre = symWays(prod[0], l, k)
		default: // intermediate prefix
			pre = countInter(slot{sl.nt, sl.alt, sl.dot - 1}, l, k)
		}
		return new(big.Int).Mul(pre, last)
	}

	countInter = func(sl slot, l, r int) *big.Int {
		id := bsrNodeID{inter: true, slot: sl, l: l, r: r}
		if v, ok := memo[id]; ok {
			return v
		}
		if onStack[id] {
			infinite = true
			return big.NewInt(1)
		}
		onStack[id] = true
		total := big.NewInt(0)
		for _, k := range f.index[bsrKey{sl, l, r}] {
			total.Add(total, countElem(sl, l, k, r))
		}
		onStack[id] = false
		memo[id] = total
		return total
	}

	countSym = func(nt, l, r int) *big.Int {
		id := bsrNodeID{nt: nt, l: l, r: r}
		if v, ok := memo[id]; ok {
			return v
		}
		if onStack[id] {
			infinite = true
			return big.NewInt(1)
		}
		onStack[id] = true
		total := big.NewInt(0)
		for ai, prod := range f.sg.nts[nt].alts {
			sl := slot{nt, ai, len(prod)}
			for _, k := range f.index[bsrKey{sl, l, r}] {
				total.Add(total, countElem(sl, l, k, r))
			}
		}
		onStack[id] = false
		memo[id] = total
		return total
	}

	res := countSym(f.start, 0, f.n)
	if infinite {
		return big.NewInt(-1)
	}
	return res
}

// Ambiguous reports whether the input has more than one distinct parse tree.
func (f *BSRForest) Ambiguous() bool {
	if !f.Valid() {
		return false
	}
	visited := map[bsrNodeID]bool{}

	var symAmb func(nt, l, r int) bool
	var interAmb func(sl slot, l, r int) bool
	var elemAmb func(sl slot, l, k, r int) bool

	symOf := func(s ssym, a, b int) bool {
		if s.kind == symNonterm {
			return symAmb(s.nt, a, b)
		}
		return false
	}
	elemAmb = func(sl slot, l, k, r int) bool {
		prod := f.sg.nts[sl.nt].alts[sl.alt]
		if symOf(prod[sl.dot-1], k, r) {
			return true
		}
		switch sl.dot {
		case 1:
			return false
		case 2:
			return symOf(prod[0], l, k)
		default:
			return interAmb(slot{sl.nt, sl.alt, sl.dot - 1}, l, k)
		}
	}
	interAmb = func(sl slot, l, r int) bool {
		id := bsrNodeID{inter: true, slot: sl, l: l, r: r}
		if visited[id] {
			return false
		}
		visited[id] = true
		ks := f.index[bsrKey{sl, l, r}]
		if len(ks) > 1 {
			return true
		}
		for _, k := range ks {
			if elemAmb(sl, l, k, r) {
				return true
			}
		}
		return false
	}
	symAmb = func(nt, l, r int) bool {
		id := bsrNodeID{nt: nt, l: l, r: r}
		if visited[id] {
			return false
		}
		visited[id] = true
		packs := 0
		for ai, prod := range f.sg.nts[nt].alts {
			sl := slot{nt, ai, len(prod)}
			ks := f.index[bsrKey{sl, l, r}]
			packs += len(ks)
		}
		if packs > 1 {
			return true
		}
		for ai, prod := range f.sg.nts[nt].alts {
			sl := slot{nt, ai, len(prod)}
			for _, k := range f.index[bsrKey{sl, l, r}] {
				if elemAmb(sl, l, k, r) {
					return true
				}
			}
		}
		return false
	}
	return symAmb(f.start, 0, f.n)
}

// Tree extracts a single parse tree (first element at each extent), or nil if
// the input is invalid. Synthetic nonterminals (groups, options, repetitions)
// are collapsed so only original rule names appear, matching Forest.Tree.
func (f *BSRForest) Tree() *ParseTree {
	if !f.Valid() {
		return nil
	}
	sl, ks, ok := f.startElems()
	if !ok {
		return nil
	}
	visited := map[bsrNodeID]bool{}
	return f.emitSym(f.start, sl, ks[0], 0, f.n, visited)
}

// emitSym builds a ParseTree node for nonterminal nt over [l,r], chosen via the
// complete-production slot sl with pivot k.
func (f *BSRForest) emitSym(nt int, sl slot, k, l, r int, visited map[bsrNodeID]bool) *ParseTree {
	var kids []*ParseTree
	f.collectElem(sl, l, k, r, &kids, visited)
	info := f.sg.nts[nt]
	if !info.isRule {
		// synthetic (group/option/rep): splice children into the parent
		return &ParseTree{Start: l, End: r, Children: kids} // caller flattens
	}
	return &ParseTree{Rule: info.ruleName, Start: l, End: r, Children: kids}
}

// collectElem appends the children of element (sl,l,k,r) into out, flattening
// intermediate extents and synthetic nonterminals.
func (f *BSRForest) collectElem(sl slot, l, k, r int, out *[]*ParseTree, visited map[bsrNodeID]bool) {
	prod := f.sg.nts[sl.nt].alts[sl.alt]
	// prefix x_1..x_{dot-1} over [l,k]
	switch sl.dot {
	case 1:
		// empty
	case 2:
		f.collectChild(prod[0], l, k, out, visited)
	default:
		isl := slot{sl.nt, sl.alt, sl.dot - 1}
		if ks := f.index[bsrKey{isl, l, k}]; len(ks) > 0 {
			f.collectElem(isl, l, ks[0], k, out, visited)
		}
	}
	// last symbol x_{dot} over [k,r]
	f.collectChild(prod[sl.dot-1], k, r, out, visited)
}

func (f *BSRForest) collectChild(s ssym, a, b int, out *[]*ParseTree, visited map[bsrNodeID]bool) {
	if s.kind != symNonterm {
		if s.kind == symEps {
			return // epsilon contributes no leaf
		}
		*out = append(*out, &ParseTree{Start: a, End: b})
		return
	}
	id := bsrNodeID{nt: s.nt, l: a, r: b}
	if visited[id] {
		return
	}
	visited[id] = true
	// pick the first complete-production element for this nonterminal extent
	for ai, prod := range f.sg.nts[s.nt].alts {
		csl := slot{s.nt, ai, len(prod)}
		if ks := f.index[bsrKey{csl, a, b}]; len(ks) > 0 {
			info := f.sg.nts[s.nt]
			if info.isRule {
				var kids []*ParseTree
				f.collectElem(csl, a, ks[0], b, &kids, visited)
				*out = append(*out, &ParseTree{Rule: info.ruleName, Start: a, End: b, Children: kids})
			} else {
				// synthetic: splice its children directly into the parent
				f.collectElem(csl, a, ks[0], b, out, visited)
			}
			visited[id] = false
			return
		}
	}
	visited[id] = false
}

func newBSRParser(sg *slotGrammar, input []byte, maxElems int) *bsrParser {
	return &bsrParser{
		sg:       sg,
		input:    input,
		set:      map[bsrElem]bool{},
		gss:      map[gssKey]*bsrGNode{},
		seen:     map[bsrDesc]bool{},
		popped:   map[*bsrGNode]map[int]bool{},
		maxElems: maxElems,
	}
}

// ParseBSR parses input under rootRulename using the BSR engine. It accepts the
// same options as ParseForest (WithMaxForestNodes bounds the BSR set size).
func ParseBSR(input []byte, grammar *Grammar, rootRulename string, opts ...ForestOption) (*BSRForest, error) {
	cfg := forestConfig{maxSlots: defaultMaxSlots}
	for _, o := range opts {
		o(&cfg)
	}
	sg, err := compileSlots(grammar, rootRulename, cfg.maxSlots)
	if err != nil {
		return nil, err
	}
	p := newBSRParser(sg, input, cfg.maxNodes)
	p.parse()
	if p.aborted {
		return nil, &ErrForestTooLarge{Max: cfg.maxNodes}
	}
	f := &BSRForest{
		sg:       sg,
		input:    input,
		rulename: rootRulename,
		set:      p.set,
		start:    sg.start,
		n:        len(input),
	}
	f.buildIndex()
	return f, nil
}
