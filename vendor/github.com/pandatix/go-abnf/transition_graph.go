package goabnf

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"slices"
	"sort"
	"strings"
	"unicode/utf8"
)

// TransitionGraph represent a grammar transition graph.
// It contains the list of entrypoints and endpoints.
// You could travel through the graph starting from the entrypoints.
type TransitionGraph struct {
	grammar  *Grammar
	rulename string
	options  *tgoptions

	Entrypoints []*Node
	Endpoints   []*Node
}

// Node of a TransitionGraph.
// Unically identifiate the node of the transition graph along
// with its underlying element and its next nodes.
type Node struct {
	ID   string
	Elem ElemItf

	Nexts []*Node
}

// TransitionGraph builds a transition graph out of a grammar
// and the given rulename.
// TODO it is possible to build transition graph out of cylic rules iff	it is not concatenated to another repetition (can't pipe O->I as there is no O). For instance, `a = "a" a` can exist.
func (g *Grammar) TransitionGraph(rulename string, opts ...TGOption) (*TransitionGraph, error) {
	// Build transition graph machine
	options := &tgoptions{
		deflateRules:        false,
		deflateNumVals:      false,
		deflateCharVals:     false,
		repetitionThreshold: 256,
		maxNodes:            0, // 0 == unbounded (backward compatible)
	}
	for _, opt := range opts {
		opt.apply(options)
	}
	m := &tgmachine{
		options: options,
		grammar: g,
		buf:     map[string][2][]*Node{},
	}

	// Find the rule
	rule := GetRule(rulename, g.Rulemap)
	if rule == nil {
		return nil, &ErrRuleNotFound{
			Rulename: rulename,
		}
	}

	// Ensure the rule is semantically valid and does not contain cycle.
	// Semantic validity is required to ensure rule dependencies actually
	// exist, while no cycle is required because the algorithm is recursive
	// and may not stop else.
	if options.deflateRules {
		if err := SemvalABNF(g); err != nil {
			return nil, err
		}
	}
	cycle, err := g.RuleContainsCycle(rulename)
	if err != nil {
		return nil, err
	}
	if cycle {
		return nil, &ErrCyclicRule{
			Rulename: rulename,
		}
	}

	// Build transition graph
	entrypoints, endpoints, err := m.altGraph(rule.Alternation)
	if err != nil {
		return nil, err
	}
	return &TransitionGraph{
		grammar:     g,
		rulename:    rulename,
		options:     options,
		Entrypoints: entrypoints,
		Endpoints:   endpoints,
	}, nil
}

var (
	// The empty node enable to bridge the no-entrypoint case of a repetition.
	// This occurs when the minimum is 0.
	// It disappear as soon as the repetition is piped as an input from lower
	// inputs to upper outputs.
	emptyNode = (*Node)(nil)
)

type tgoptions struct {
	deflateRules        bool
	deflateNumVals      bool
	deflateCharVals     bool
	repetitionThreshold int
	maxNodes            int
}

type TGOption interface {
	apply(opts *tgoptions)
}

type deflateRulesOption bool

func (opt deflateRulesOption) apply(opts *tgoptions) {
	opts.deflateRules = bool(opt)
}

// WithDeflateRules when passed to true, will recursively build the transition
// graphs of the rules it uses (reuse them if possible) and concatenate the
// sub transition graphs.
// It generates a more complex but exhaustive transition graph.
func WithDeflateRules(deflate bool) TGOption {
	return deflateRulesOption(deflate)
}

var _ TGOption = deflateRulesOption(false)

type deflateNumValsOption bool

func (opt deflateNumValsOption) apply(opts *tgoptions) {
	opts.deflateNumVals = bool(opt)
}

// WithDeflateNumVals when passed to true, will replace a single node representing
// the ranges or series of multiple numeric values to individual nodes.
// It generates a more complex but exhaustive transition graph.
func WithDeflateNumVals(deflate bool) TGOption {
	return deflateNumValsOption(deflate)
}

var _ TGOption = deflateNumValsOption(false)

type deflateCharVals bool

func (opt deflateCharVals) apply(opts *tgoptions) {
	opts.deflateCharVals = bool(opt)
}

// WithDeflateCharVals when passed to true, will replace a single node representing
// the char value to an exhaustive concatenation of all characters (lower and upper
// case) and the transitions between them.
func WithDeflateCharVals(deflate bool) TGOption {
	return deflateCharVals(deflate)
}

var _ TGOption = deflateCharVals(false)

type repetitionThreshold int

func (opt repetitionThreshold) apply(opts *tgoptions) {
	opts.repetitionThreshold = int(opt)
}

// WithRepetitionThreshold defines the threshold to block a large repetition
// to occur, elseway it may consum all memory to build the transition graph.
// For instance, a rule defined as follows may self-DoS go-abnf.
//
// a = 9555("a")
//
// WARNING it does not avoid such memory consumption on chained repetitions.
// For instance, the two following cases could produce large transition graphs
// without control with this functional option.
//
// a = 10( 10( 10("a") ) )
//
// a = 10 ( 10*"a" ( 10"a" 9"a" 8"a" ) / ( *10"a" *10"a" *10"a" ) ) 10"a"
//
// Defaults to 256.
func WithRepetitionThreshold(threshold int) TGOption {
	return repetitionThreshold(threshold)
}

type maxNodesOption int

func (opt maxNodesOption) apply(opts *tgoptions) {
	opts.maxNodes = int(opt)
}

// WithMaxNodes bounds the total number of nodes the transition graph may
// allocate during construction. Unlike WithRepetitionThreshold, which only caps
// a single repetition, this bounds the *whole* graph and so also defends against
// the combinatorial blow-up of nested repetitions (e.g. 10( 10( 10("a") ) )) and
// of wide num-val ranges deflated with WithDeflateNumVals. When the bound would
// be exceeded, construction stops with *ErrMaxNodesExceeded before the nodes are
// allocated.
//
// A value <= 0 (the default) means unbounded, preserving the previous behaviour.
// Set a positive bound when building transition graphs from untrusted grammars
// to keep the work within a controlled domain.
func WithMaxNodes(max int) TGOption {
	return maxNodesOption(max)
}

// reserve accounts for n nodes about to be created, failing fast (before any
// allocation) when the WithMaxNodes budget would be exceeded. It is a no-op when
// the budget is unbounded.
func (m *tgmachine) reserve(n int) error {
	if m.options.maxNodes <= 0 {
		return nil // unbounded
	}
	if n < 0 || m.nodeCount+n < 0 || m.nodeCount+n > m.options.maxNodes {
		return &ErrMaxNodesExceeded{Max: m.options.maxNodes}
	}
	m.nodeCount += n
	return nil
}

// reserveClones accounts for cloning the subgraph reachable from entries reps
// times. When the budget is unbounded it does no work at all (in particular it
// skips the reachability count), so the default path keeps its previous cost.
func (m *tgmachine) reserveClones(reps int, entries []*Node) error {
	if m.options.maxNodes <= 0 {
		return nil
	}
	return m.reserve(reps * countReachable(entries))
}

// countReachable counts the distinct non-empty nodes reachable from entries,
// i.e. how many nodes cloneTransitionGraph would clone.
func countReachable(entries []*Node) int {
	seen := map[string]bool{}
	stack := make([]*Node, 0, len(entries))
	for _, e := range entries {
		if e != emptyNode {
			stack = append(stack, e)
		}
	}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if n == emptyNode || seen[n.ID] {
			continue
		}
		seen[n.ID] = true
		for _, sc := range n.Nexts {
			if sc != emptyNode && !seen[sc.ID] {
				stack = append(stack, sc)
			}
		}
	}
	return len(seen)
}

// CoverageMode selects what set of strings a TransitionGraphReader produces.
type CoverageMode int

const (
	// CoverageAllTrails (mode A, the default) emits every entrypoint->endpoint
	// trail (a walk using each directed edge at most once) and, for each trail,
	// every combination of its nodes' terminal variations. It is finite and
	// terminating for any graph, but on dense graphs or wide range/charval nodes
	// the number of trail x variation combinations can be very large.
	CoverageAllTrails CoverageMode = iota

	// CoverageCompact (mode B) emits a small set of strings that still covers
	// every reachable vertex, every edge, and every terminal value (each charval
	// casing, each numval range value) at least once, then stops. Its size is
	// bounded by roughly (#edges + #vertices + sum of per-node variations), so it
	// stays small even for grammars where CoverageAllTrails explodes.
	CoverageCompact
)

// ReaderOption configures a TransitionGraphReader.
type ReaderOption interface {
	apply(*readerOptions)
}

type readerOptions struct {
	mode CoverageMode
}

type coverageModeOption CoverageMode

func (o coverageModeOption) apply(opts *readerOptions) {
	opts.mode = CoverageMode(o)
}

// WithCoverageMode selects the coverage strategy of the reader (CoverageAllTrails
// by default, or CoverageCompact for a bounded covering set).
func WithCoverageMode(mode CoverageMode) ReaderOption {
	return coverageModeOption(mode)
}

func (tg *TransitionGraph) Reader(opts ...ReaderOption) *TransitionGraphReader {
	ro := readerOptions{mode: CoverageAllTrails}
	for _, o := range opts {
		o.apply(&ro)
	}
	return &TransitionGraphReader{tg: tg, mode: ro.mode}
}

// TransitionGraphReader produces a finite set of strings that COVERS the
// transition graph: every reachable vertex and edge that lies on some
// entrypoint->endpoint walk appears in at least one produced string. It does so
// by enumerating every entrypoint->endpoint *trail* (a walk that uses each
// directed edge at most once) in order of increasing length, and for each trail
// every combination of its nodes' terminal variations (char-val casing,
// num-val ranges). Trails are bounded by the number of edges, so the set is
// finite and enumeration always terminates -- for any graph, cyclic or not.
type TransitionGraphReader struct {
	tg *TransitionGraph

	mode CoverageMode
	all  map[string]*Node // every node reachable from the entrypoints

	// CoverageCompact precomputed output.
	prods        [][]byte
	prodIdx      int
	builtCompact bool

	prepared bool
	entries  []*Node         // entrypoints that can reach an endpoint via producing nodes
	endSet   map[string]bool // endpoint node IDs
	canReach map[string]bool // node can reach an endpoint via producing nodes

	// Trail enumeration state.
	length          int            // current trail length (number of nodes)
	stack           []pathFrame    // current trail being walked
	usedEdges       map[string]int // edges currently on the trail (>0 == used)
	levelHadAnyPath bool           // any length-`length` trail seen this level
	exhausted       bool           // no more trails (coverage complete)

	// Variation odometer for the current trail.
	havePath bool
	varIdx   []int32
	varTot   []int32

	// Staged production.
	staged    []byte
	hasStaged bool
}

type pathFrame struct {
	node    *Node
	cursor  int
	allowed []*Node // children selectable at this depth given the trail prefix
}

func tgNodeID(n *Node) string {
	if n == nil {
		return ""
	}
	return n.ID
}

// producing reports whether a node can emit terminal bytes. Rulename nodes
// (present only when rules are not deflated) cannot, so no trail may pass
// through them.
func producing(n *Node) bool {
	if n == emptyNode {
		return true
	}
	switch n.Elem.(type) {
	case ElemCharVal, ElemNumVal:
		return true
	default:
		return false
	}
}

func (tgr *TransitionGraphReader) prepare() {
	if tgr.prepared {
		return
	}
	tgr.prepared = true
	tgr.endSet = map[string]bool{}
	for _, e := range tgr.tg.Endpoints {
		tgr.endSet[tgNodeID(e)] = true
	}
	tgr.usedEdges = map[string]int{}

	// Collect every node reachable from the entrypoints.
	all := map[string]*Node{}
	var collect func(n *Node)
	collect = func(n *Node) {
		id := tgNodeID(n)
		if _, ok := all[id]; ok {
			return
		}
		all[id] = n
		if n == emptyNode {
			return // the empty sentinel ((*Node)(nil)) has no successors
		}
		for _, s := range n.Nexts {
			collect(s)
		}
	}
	for _, e := range tgr.tg.Entrypoints {
		collect(e)
	}
	tgr.all = all

	// canReach[n] = n can produce AND (n is an endpoint OR some producing
	// successor of n can reach an endpoint). Monotonic fixpoint.
	tgr.canReach = map[string]bool{}
	for changed := true; changed; {
		changed = false
		for id, n := range all {
			if tgr.canReach[id] || !producing(n) {
				continue
			}
			ok := tgr.endSet[id]
			if !ok && n != emptyNode {
				for _, sc := range n.Nexts {
					if producing(sc) && tgr.canReach[tgNodeID(sc)] {
						ok = true
						break
					}
				}
			}
			if ok {
				tgr.canReach[id] = true
				changed = true
			}
		}
	}

	for _, e := range tgr.tg.Entrypoints {
		if producing(e) && tgr.canReach[tgNodeID(e)] {
			tgr.entries = append(tgr.entries, e)
		}
	}
}

func edgeStr(from, to *Node) string {
	return tgNodeID(from) + ">" + tgNodeID(to)
}

// allowedAt returns the children selectable at depth d given the current trail
// prefix: successors that can still reach an endpoint and whose edge is not yet
// used on this trail. At depth 0 the "children" are the entrypoints.
func (tgr *TransitionGraphReader) allowedAt(d int) []*Node {
	if d == 0 {
		return tgr.entries
	}
	parent := tgr.stack[d-1].node
	if parent == emptyNode {
		return nil // empty sentinel has no successors
	}
	var out []*Node
	for _, sc := range parent.Nexts {
		if !tgr.canReach[tgNodeID(sc)] {
			continue
		}
		if tgr.usedEdges[edgeStr(parent, sc)] > 0 {
			continue
		}
		out = append(out, sc)
	}
	return out
}

func (tgr *TransitionGraphReader) pushFirst(d int) bool {
	allowed := tgr.allowedAt(d)
	if len(allowed) == 0 {
		return false
	}
	node := allowed[0]
	if d >= 1 {
		tgr.usedEdges[edgeStr(tgr.stack[d-1].node, node)]++
	}
	tgr.stack = append(tgr.stack, pathFrame{node: node, cursor: 0, allowed: allowed})
	return true
}

func (tgr *TransitionGraphReader) bump() bool {
	for len(tgr.stack) > 0 {
		d := len(tgr.stack) - 1
		f := &tgr.stack[d]
		if d >= 1 {
			tgr.usedEdges[edgeStr(tgr.stack[d-1].node, f.node)]--
		}
		if f.cursor+1 < len(f.allowed) {
			f.cursor++
			f.node = f.allowed[f.cursor]
			if d >= 1 {
				tgr.usedEdges[edgeStr(tgr.stack[d-1].node, f.node)]++
			}
			return true
		}
		tgr.stack = tgr.stack[:d]
	}
	return false
}

// nextLeaf advances the trail to the next walk of exactly tgr.length nodes.
func (tgr *TransitionGraphReader) nextLeaf() bool {
	L := tgr.length
	for {
		switch {
		case len(tgr.stack) == L:
			if !tgr.bump() {
				return false
			}
		case len(tgr.stack) < L:
			if !tgr.pushFirst(len(tgr.stack)) {
				if !tgr.bump() {
					return false
				}
			}
		}
		if len(tgr.stack) == L {
			tgr.levelHadAnyPath = true
			return true
		}
	}
}

func (tgr *TransitionGraphReader) resetLevel() {
	tgr.stack = tgr.stack[:0]
	for k := range tgr.usedEdges {
		delete(tgr.usedEdges, k)
	}
	tgr.levelHadAnyPath = false
}

// advancePath sets tgr.stack to the next entrypoint->endpoint trail, returning
// false once every trail has been produced (coverage complete).
func (tgr *TransitionGraphReader) advancePath() bool {
	for {
		if tgr.length == 0 {
			tgr.length = 1
			tgr.resetLevel()
		}
		for tgr.nextLeaf() {
			if tgr.endSet[tgNodeID(tgr.stack[len(tgr.stack)-1].node)] {
				return true
			}
		}
		if !tgr.levelHadAnyPath {
			return false // no trail this long exists => done
		}
		tgr.length++
		tgr.resetLevel()
	}
}

func (tgr *TransitionGraphReader) ensureStaged() {
	if tgr.hasStaged || tgr.exhausted {
		return
	}
	if !tgr.havePath {
		if !tgr.advancePath() {
			tgr.exhausted = true
			return
		}
		tgr.havePath = true
		tgr.varTot = tgr.varTot[:0]
		tgr.varIdx = tgr.varIdx[:0]
		for i := range tgr.stack {
			_, vt := nodeEmit(tgr.stack[i].node, 0)
			if vt < 1 {
				vt = 1
			}
			tgr.varTot = append(tgr.varTot, vt)
			tgr.varIdx = append(tgr.varIdx, 0)
		}
	}
	tgr.staged = tgr.staged[:0]
	for i := range tgr.stack {
		b, _ := nodeEmit(tgr.stack[i].node, tgr.varIdx[i])
		tgr.staged = append(tgr.staged, b...)
	}
	tgr.hasStaged = true
}

func (tgr *TransitionGraphReader) consumeStaged() {
	tgr.hasStaged = false
	i := len(tgr.varIdx) - 1
	for i >= 0 {
		tgr.varIdx[i]++
		if tgr.varIdx[i] < tgr.varTot[i] {
			break
		}
		tgr.varIdx[i] = 0
		i--
	}
	if i < 0 {
		tgr.havePath = false // variations exhausted -> move to next trail
	}
}

// Next reports whether another covering string is available.
func (tgr *TransitionGraphReader) Next() bool {
	tgr.prepare()
	if tgr.mode == CoverageCompact {
		tgr.buildCompact()
		return tgr.prodIdx < len(tgr.prods)
	}
	tgr.ensureStaged()
	return tgr.hasStaged
}

// Scan returns the next covering string, or nil once coverage is complete.
func (tgr *TransitionGraphReader) Scan() []byte {
	tgr.prepare()
	if tgr.mode == CoverageCompact {
		tgr.buildCompact()
		if tgr.prodIdx >= len(tgr.prods) {
			return nil
		}
		out := tgr.prods[tgr.prodIdx]
		tgr.prodIdx++
		return out
	}
	tgr.ensureStaged()
	if !tgr.hasStaged {
		return nil
	}
	out := make([]byte, len(tgr.staged))
	copy(out, tgr.staged)
	tgr.consumeStaged()
	return out
}

// buildCompact precomputes the CoverageCompact output: a bounded set of strings
// that together visit every reachable vertex, traverse every edge, and emit every
// terminal value (each charval casing, each numval range value) at least once.
// It is greedy, not minimal, but its size is bounded by roughly
// (#edges + #vertices + sum of per-node variation counts).
func (tgr *TransitionGraphReader) buildCompact() {
	if tgr.builtCompact {
		return
	}
	tgr.builtCompact = true

	viable := func(n *Node) []*Node {
		if n == emptyNode {
			return nil
		}
		var out []*Node
		for _, sc := range n.Nexts {
			if producing(sc) && tgr.canReach[tgNodeID(sc)] {
				out = append(out, sc)
			}
		}
		return out
	}

	// Forward BFS from entrypoints: coverable vertices, edges, and entry->node
	// shortest-path predecessors.
	predEntry := map[string]*Node{}
	nodeByID := map[string]*Node{}
	discOrder := map[string]int{} // stable BFS discovery index per node
	var disc []*Node              // nodes in discovery order
	type edge struct{ u, w *Node }
	var allEdges []edge
	seenEdge := map[string]bool{}
	var queue []*Node
	see := func(n *Node) {
		id := tgNodeID(n)
		if _, ok := predEntry[id]; ok {
			return
		}
		predEntry[id] = n
		nodeByID[id] = n
		discOrder[id] = len(disc)
		disc = append(disc, n)
		queue = append(queue, n)
	}
	for _, e := range tgr.entries {
		see(e)
	}
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		for _, sc := range viable(n) {
			sid := tgNodeID(sc)
			ek := tgNodeID(n) + ">" + sid
			if !seenEdge[ek] {
				seenEdge[ek] = true
				allEdges = append(allEdges, edge{n, sc})
			}
			if _, ok := predEntry[sid]; !ok {
				predEntry[sid] = n
				nodeByID[sid] = sc
				discOrder[sid] = len(disc)
				disc = append(disc, sc)
				queue = append(queue, sc)
			}
		}
	}

	// Backward BFS from endpoints: node->endpoint shortest-path successors.
	// Node IDs are random UUIDs, so we order everything by stable BFS discovery
	// index instead -- otherwise the chosen shortest paths (and thus the covering
	// set) would vary between builds.
	revAdj := map[string][]*Node{}
	for _, n := range disc {
		for _, sc := range viable(n) {
			sid := tgNodeID(sc)
			revAdj[sid] = append(revAdj[sid], n)
		}
	}
	succEnd := map[string]*Node{}
	var q2 []*Node
	for _, n := range disc {
		if tgr.endSet[tgNodeID(n)] {
			succEnd[tgNodeID(n)] = n
			q2 = append(q2, n)
		}
	}
	for len(q2) > 0 {
		n := q2[0]
		q2 = q2[1:]
		for _, p := range revAdj[tgNodeID(n)] {
			pid := tgNodeID(p)
			if _, ok := succEnd[pid]; !ok {
				succEnd[pid] = n
				q2 = append(q2, p)
			}
		}
	}

	entryPath := func(u *Node) []*Node {
		var rev []*Node
		cur := u
		for {
			rev = append(rev, cur)
			id := tgNodeID(cur)
			p := predEntry[id]
			if tgNodeID(p) == id {
				break
			}
			cur = p
		}
		for i, j := 0, len(rev)-1; i < j; i, j = i+1, j-1 {
			rev[i], rev[j] = rev[j], rev[i]
		}
		return rev
	}
	endPath := func(w *Node) []*Node {
		var out []*Node
		cur := w
		for {
			out = append(out, cur)
			id := tgNodeID(cur)
			sc := succEnd[id]
			if tgNodeID(sc) == id {
				break
			}
			cur = sc
		}
		return out
	}

	coveredV := map[string]bool{}
	coveredE := map[string]bool{}
	// nextVal[id] is the smallest variation index of a node not yet emitted.
	// Values are always taken in increasing order, so a forward cursor per node
	// makes value coverage O(total values) instead of O(sum of width^2).
	nextVal := map[string]int32{}

	markWalk := func(w []*Node) {
		for i, n := range w {
			coveredV[tgNodeID(n)] = true
			if i+1 < len(w) {
				coveredE[tgNodeID(n)+">"+tgNodeID(w[i+1])] = true
			}
		}
	}
	emit := func(w []*Node, force bool) bool {
		va := make([]int32, len(w))
		claimed := map[string]int32{} // fresh values taken for each node in THIS emit
		newCount := 0
		for p, n := range w {
			id := tgNodeID(n)
			_, vt := nodeEmit(n, 0)
			if vt < 1 {
				vt = 1
			}
			claim := nextVal[id] + claimed[id]
			if claim < vt {
				va[p] = claim
				claimed[id]++
				newCount++
			} else {
				va[p] = 0 // all of this node's values already covered; reuse 0
			}
		}
		if !force && newCount == 0 {
			return false
		}
		for id, c := range claimed {
			nextVal[id] += c
		}
		var bs []byte
		for p, n := range w {
			b, _ := nodeEmit(n, va[p])
			bs = append(bs, b...)
		}
		tgr.prods = append(tgr.prods, bs)
		return true
	}

	var walks [][]*Node

	sort.SliceStable(allEdges, func(i, j int) bool {
		ui, uj := discOrder[tgNodeID(allEdges[i].u)], discOrder[tgNodeID(allEdges[j].u)]
		if ui != uj {
			return ui < uj
		}
		return discOrder[tgNodeID(allEdges[i].w)] < discOrder[tgNodeID(allEdges[j].w)]
	})
	for _, e := range allEdges {
		ek := tgNodeID(e.u) + ">" + tgNodeID(e.w)
		if coveredE[ek] {
			continue
		}
		if _, ok := succEnd[tgNodeID(e.w)]; !ok {
			continue
		}
		w := append(entryPath(e.u), endPath(e.w)...)
		walks = append(walks, w)
		emit(w, true)
		markWalk(w)
	}

	for _, v := range disc {
		id := tgNodeID(v)
		if coveredV[id] {
			continue
		}
		if _, ok := succEnd[id]; !ok {
			continue
		}
		var w []*Node
		if tgr.endSet[id] {
			w = entryPath(v)
		} else {
			ep := endPath(v)
			w = append(entryPath(v), ep[1:]...)
		}
		walks = append(walks, w)
		emit(w, true)
		markWalk(w)
	}

	for _, w := range walks {
		for emit(w, false) {
		}
	}
}

// nodeEmit returns the bytes a node emits for variation index tpos, and the
// node's total number of variations. For a non-sensitive char-val the variations
// are the 2^k casings of its k cased letters; for a num-val range, one byte per
// value in the range.
func nodeEmit(node *Node, tpos int32) (prod []byte, vtotal int32) {
	if node == emptyNode {
		return nil, 1
	}
	switch v := node.Elem.(type) {
	case ElemCharVal:
		if v.Sensitive {
			for _, val := range v.Values {
				prod = append(prod, string(val)...)
			}
			return prod, 1
		}
		var numVar int
		for _, r := range v.Values {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
				numVar++
			}
		}
		// 2^numVar casings; cap to avoid int32 overflow on pathologically long
		// case-insensitive literals (2^31 casings is unenumerable anyway). Letters
		// beyond the cap are emitted lower-case only.
		// XXX This is an arbitrary bound
		if numVar > 30 {
			numVar = 30
		}
		vtotal = int32(1) << uint(numVar)
		bit := uint(0)
		for _, r := range v.Values {
			isLower := r >= 'a' && r <= 'z'
			isUpper := r >= 'A' && r <= 'Z'
			if !isLower && !isUpper {
				prod = append(prod, string(r)...)
				continue
			}
			lower := r
			if isUpper {
				lower = lower - 'A' + 'a'
			}
			upper := r
			if isLower {
				upper = upper - 'a' + 'A'
			}
			if (tpos>>bit)&1 == 1 {
				prod = append(prod, string(upper)...)
			} else {
				prod = append(prod, string(lower)...)
			}
			bit++
		}
		return prod, vtotal

	case ElemNumVal:
		switch v.Status {
		case StatSeries:
			for _, elem := range v.Elems {
				r := numvalToRune(elem, v.Base)
				prod = append(prod, []byte(string(r))...)
			}
			return prod, 1
		case StatRange:
			min, max := numvalToInt32(v.Elems[0], v.Base), numvalToInt32(v.Elems[1], v.Base)
			// The graph operates over runes; clamp to the Unicode range so an
			// out-of-range bound cannot overflow the int32 count below.
			if max > utf8.MaxRune {
				max = utf8.MaxRune
			}
			if min > utf8.MaxRune || min > max {
				return nil, 1 // no representable character in this range
			}
			r := rune(min + tpos)
			return []byte(string(r)), max - min + 1
		}
	}
	return nil, 1
}

type tgmachine struct {
	options *tgoptions
	grammar *Grammar
	buf     map[string][2][]*Node

	nodeCount int // nodes reserved so far, checked against options.maxNodes
}

func (m *tgmachine) altGraph(alt Alternation) (entrypoints []*Node, endpoints []*Node, err error) {
	for _, concat := range alt.Concatenations {
		ci, co, err := m.concatGraph(concat)
		if err != nil {
			return nil, nil, err
		}

		// Group concatenation inputs, but avoid duplicating the emptyNode
		entryHasEmpty := slices.Contains(entrypoints, emptyNode)
		for _, vci := range ci {
			if vci == emptyNode && entryHasEmpty {
				continue
			}
			entrypoints = append(entrypoints, vci)
		}

		// Group concatenation outputs, but avoid duplicating the emptyNode
		endHasEmpty := slices.Contains(endpoints, emptyNode)
		for _, vco := range co {
			if vco == emptyNode && endHasEmpty {
				continue
			}
			endpoints = append(endpoints, vco)
		}
	}
	return
}

func (m *tgmachine) concatGraph(concat Concatenation) (entrypoints []*Node, endpoints []*Node, err error) {
	ci, co, err := m.repGraph(concat.Repetitions[0])
	if err != nil {
		return nil, nil, err
	}
	for i := 1; i < len(concat.Repetitions); i++ {
		repi, repo, err := m.repGraph(concat.Repetitions[i])
		if err != nil {
			return nil, nil, err
		}
		ci, co = concatTransitionGraphs(ci, co, repi, repo)
	}
	return ci, co, nil
}

func concatTransitionGraphs(prevI, prevO, currI, currO []*Node) (entrypoints []*Node, endpoints []*Node) {
	// If prevI contains the emptyNode it means currI (possibly with
	// the emptyNode) is also part of the entrypoints (without prevI's
	// emptyNode).
	if slices.Contains(prevI, emptyNode) {
		for _, vprevI := range prevI {
			if vprevI == emptyNode {
				continue
			}
			entrypoints = append(entrypoints, vprevI)
		}
		entrypoints = append(entrypoints, currI...)
	} else {
		entrypoints = prevI
	}

	// plug previous outputs (prevO) to current inputs (currI).
	for _, vprevO := range prevO {
		if vprevO == emptyNode {
			continue
		}
		for _, vrepi := range currI {
			if vrepi == emptyNode {
				continue
			}
			vprevO.Nexts = append(vprevO.Nexts, vrepi)
		}
	}

	// new concatenation output is the previous ones if current
	// inputs contains the emptyNode (repetition is skippable),
	// and the current outputs ones.
	if slices.Contains(currI, emptyNode) {
		endpoints = append(endpoints, prevO...)
	}
	for _, vrepo := range currO {
		if vrepo == emptyNode {
			continue
		}
		endpoints = append(endpoints, vrepo)
	}

	return
}

func (m *tgmachine) repGraph(rep Repetition) (entrypoints []*Node, endpoints []*Node, err error) {
	// element entry-/end-points (resp. I/O)
	elemi, elemo, err := m.elemGraph(rep.Element)
	if err != nil {
		return nil, nil, err
	}

	if rep.Min == 0 {
		entrypoints = appendNodes(entrypoints, emptyNode)
		endpoints = appendNodes(endpoints, emptyNode)

		switch rep.Max {
		case 1:
			// min = 0 & max = 1: optional v
			entrypoints = appendNodes(entrypoints, elemi...)
			endpoints = appendNodes(endpoints, elemo...)
		case inf:
			// min = 0 & max = inf: optional infinity
			elemiNoEmpty := []*Node{}
			for _, velemi := range elemi {
				if velemi == emptyNode {
					continue
				}
				elemiNoEmpty = appendNodes(elemiNoEmpty, velemi)
			}
			for _, velemo := range elemo {
				if velemo == emptyNode {
					continue
				}
				velemo.Nexts = appendNodes(velemo.Nexts, elemiNoEmpty...)
			}
			entrypoints = appendNodes(entrypoints, elemi...)
			endpoints = appendNodes(endpoints, elemo...)

		default:
			// min = 0 & 1 < max < inf: optional to n
			if rep.Max > m.options.repetitionThreshold {
				return nil, nil, ErrRepetitionThreshold
			}
			if err := m.reserveClones(rep.Max, elemi); err != nil {
				return nil, nil, err
			}
			tgs, chi, _ := chainTransitionGraph(elemi, elemo, rep.Max)
			entrypoints = appendNodes(entrypoints, chi...)
			for _, tg := range tgs {
				endpoints = appendNodes(endpoints, tg.Endpoints...)
			}
		}
	} else {
		switch rep.Max {
		case 1:
			// min == 1 & max == 1: mandatory
			if rep.Min != 1 {
				return nil, nil, ErrRepetitionMinNotOne
			}
			entrypoints = appendNodes(entrypoints, elemi...)
			endpoints = appendNodes(endpoints, elemo...)

		case inf:
			// min >= 1 && max = inf: n to infinity
			if rep.Min > m.options.repetitionThreshold {
				return nil, nil, ErrRepetitionThreshold
			}
			if err := m.reserveClones(rep.Min, elemi); err != nil {
				return nil, nil, err
			}
			tgs, chi, cho := chainTransitionGraph(elemi, elemo, rep.Min)
			entrypoints = appendNodes(entrypoints, chi...)
			endpoints = appendNodes(endpoints, cho...)

			last := tgs[len(tgs)-1]
			for _, vo := range last.Endpoints {
				if vo == emptyNode {
					continue
				}
				for _, etp := range last.Entrypoints {
					// Make sure to not append the empty node as a next node, this is not possible
					if etp == emptyNode {
						continue
					}
					vo.Nexts = appendNodes(vo.Nexts, etp)
				}
			}

		default:
			// min >= 1 && max < inf: n to m
			if rep.Min > rep.Max {
				return nil, nil, ErrRepetitionMinGtMax
			}
			if rep.Max > m.options.repetitionThreshold {
				return nil, nil, ErrRepetitionThreshold
			}

			// Build the flat chain (not endpoints, except last)
			firstI, lastO := elemi, elemo
			if rep.Min > 1 {
				if err := m.reserveClones(rep.Min, elemi); err != nil {
					return nil, nil, err
				}
				_, firstI, lastO = chainTransitionGraph(elemi, elemo, rep.Min)
			}
			entrypoints = appendNodes(entrypoints, firstI...)
			endpoints = appendNodes(endpoints, lastO...)

			// Build remaining endpoints
			remaining := rep.Max - rep.Min
			if remaining > 0 {
				if err := m.reserveClones(rep.Max-rep.Min, elemi); err != nil {
					return nil, nil, err
				}
				tgs, chi, _ := chainTransitionGraph(elemi, elemo, rep.Max-rep.Min)
				for _, tg := range tgs {
					endpoints = appendNodes(endpoints, tg.Endpoints...)
				}
				for _, vlasto := range lastO {
					if vlasto == emptyNode {
						continue
					}
					for _, nchi := range chi {
						if nchi == emptyNode {
							continue
						}
						vlasto.Nexts = appendNodes(vlasto.Nexts, nchi)
					}
				}
			}
		}
	}

	return
}

func chainTransitionGraph(i, o []*Node, n int) (tgs []TransitionGraph, entrypoints []*Node, endpoints []*Node) {
	// Make sure to not chain... nothing ¯\_(ツ)_/¯
	if n == 0 {
		return
	}
	tgs = make([]TransitionGraph, n)

	// Init chaining
	entrypoints, endpoints = cloneTransitionGraph(i, o)
	tgs[0] = TransitionGraph{
		Entrypoints: entrypoints,
		Endpoints:   endpoints,
	}

	// Gotta chain them all
	for j := 1; j < n; j++ {
		nextI, nextO := cloneTransitionGraph(i, o)
		tgs[j] = TransitionGraph{
			Entrypoints: nextI,
			Endpoints:   nextO,
		}
		entrypoints, endpoints = concatTransitionGraphs(entrypoints, endpoints, nextI, nextO)
	}
	return
}

func cloneTransitionGraph(i, o []*Node) (entrypoints []*Node, endpoints []*Node) {
	// Traverse graph to map all cloned originToNewNodes by their origin IDs
	originToNewNodes := map[string]cnode{}
	for _, vi := range i {
		if vi == emptyNode {
			continue
		}
		cloneTG(originToNewNodes, vi)
	}

	// Rebuild links
	for _, cnode := range originToNewNodes {
		nnxts := make([]*Node, 0, len(cnode.origin.Nexts))
		for _, onext := range cnode.origin.Nexts {
			nnxts = append(nnxts, originToNewNodes[onext.ID].cloned)
		}
		cnode.cloned.Nexts = nnxts
	}

	// Get cloned I/O
	entrypoints = make([]*Node, 0, len(i))
	for _, vi := range i {
		if vi == emptyNode {
			entrypoints = append(entrypoints, emptyNode)
			continue
		}
		entrypoints = append(entrypoints, originToNewNodes[vi.ID].cloned)
	}
	endpoints = make([]*Node, 0, len(o))
	for _, vo := range o {
		if vo == emptyNode {
			endpoints = append(endpoints, emptyNode)
			continue
		}
		endpoints = append(endpoints, originToNewNodes[vo.ID].cloned)
	}
	return
}

type cnode struct {
	origin, cloned *Node
}

func cloneTG(originToNewNodes map[string]cnode, origin *Node) {
	id := newID()

	originToNewNodes[origin.ID] = cnode{
		origin: origin,
		cloned: &Node{
			ID:    id,
			Elem:  origin.Elem,
			Nexts: nil, // will map it later, requires all origins to be cloned first
		},
	}

	// recurse iff not known yet
	for _, n := range origin.Nexts {
		if _, ok := originToNewNodes[n.ID]; ok {
			continue
		}
		cloneTG(originToNewNodes, n)
	}
}

func appendNodes(existing []*Node, news ...*Node) []*Node {
	for _, n := range news {
		if slices.Contains(existing, n) {
			continue
		}
		existing = append(existing, n)
	}
	return existing
}

func (m *tgmachine) elemGraph(elem ElemItf) (entrypoints []*Node, endpoints []*Node, err error) {
	switch v := elem.(type) {
	// Final elements => create the node, no need to pipe I/O
	case ElemCharVal:
		if len(v.Values) == 0 {
			entrypoints = append(entrypoints, emptyNode)
			endpoints = append(endpoints, emptyNode)
			return
		}

		if !m.options.deflateCharVals {
			if err := m.reserve(1); err != nil {
				return nil, nil, err
			}
			n := newNode(v)
			entrypoints = append(entrypoints, n)
			endpoints = append(endpoints, n)
			return
		}

		if err := m.reserve(2 * len(v.Values)); err != nil {
			return nil, nil, err
		}

		var prevs []*Node = nil
		var curr []*Node = nil
		for _, r := range v.Values {
			isLower := r >= 'a' && r <= 'z'
			isUpper := r >= 'A' && r <= 'Z'
			requireBoth := (isLower || isUpper) && !v.Sensitive

			if !requireBoth {
				n := newNode(ElemCharVal{
					Sensitive: true,
					Values:    []rune{r},
				})
				curr = []*Node{n}

				if prevs == nil {
					entrypoints = append(entrypoints, n)
				}
				for _, v := range prevs {
					v.Nexts = append(v.Nexts, curr...)
				}
				prevs = []*Node{n}
			} else {
				nlv := r
				if isUpper {
					nlv = runeMin(nlv)
				}
				nuv := r
				if isLower {
					nuv = runeMax(nuv)
				}

				nl := newNode(ElemCharVal{
					Sensitive: true,
					Values:    []rune{nlv},
				})
				nu := newNode(ElemCharVal{
					Sensitive: true,
					Values:    []rune{nuv},
				})
				curr = []*Node{nl, nu}

				if prevs == nil {
					entrypoints = append(entrypoints, nl, nu)
				}
				for _, v := range prevs {
					v.Nexts = append(v.Nexts, curr...)
				}
				prevs = []*Node{nl, nu}
			}
		}
		endpoints = append(endpoints, curr...)
		return

	case ElemNumVal:
		if !m.options.deflateNumVals {
			if err := m.reserve(1); err != nil {
				return nil, nil, err
			}
			n := newNode(v)
			entrypoints = append(entrypoints, n)
			endpoints = append(endpoints, n)
			return
		}
		switch v.Status {
		case StatRange:
			min, max := numvalToInt32(v.Elems[0], v.Base), numvalToInt32(v.Elems[1], v.Base)
			// One node is enumerated per rune; clamp to the Unicode range so an
			// out-of-range bound cannot overflow the count (int32) and bypass the
			// node budget, which would turn this loop into a DoS. A range wholly
			// above U+10FFFF contributes no node (it matches nothing).
			if max > utf8.MaxRune {
				max = utf8.MaxRune
			}
			if min > utf8.MaxRune || min > max {
				return
			}
			if err := m.reserve(int(max - min + 1)); err != nil {
				return nil, nil, err
			}
			for i := min; i <= max; i++ {
				n := newNode(ElemNumVal{
					Base:   v.Base,
					Status: StatSeries,
					Elems:  []string{int32ToNumval(i, v.Base)},
				})
				entrypoints = append(entrypoints, n)
				endpoints = append(endpoints, n)
			}

		case StatSeries:
			if err := m.reserve(len(v.Elems)); err != nil {
				return nil, nil, err
			}
			for _, s := range v.Elems {
				n := newNode(ElemNumVal{
					Base:   v.Base,
					Status: StatSeries,
					Elems:  []string{s},
				})
				entrypoints = append(entrypoints, n)
				endpoints = append(endpoints, n)
			}
		}
		return

	// Recursive elements => pipe I/O
	case ElemRulename:
		if m.options.deflateRules {
			name := strings.ToLower(v.Name)
			if n, ok := m.buf[name]; ok {
				if err := m.reserveClones(1, n[0]); err != nil {
					return nil, nil, err
				}
				entrypoints, endpoints := cloneTransitionGraph(n[0], n[1])
				return entrypoints, endpoints, nil
			}
			rule := GetRule(v.Name, m.grammar.Rulemap)
			if rule == nil {
				return nil, nil, &ErrRuleNotFound{
					Rulename: v.Name,
				}
			}
			i, o, err := m.altGraph(rule.Alternation)
			if err != nil {
				return nil, nil, err
			}
			if err := m.reserveClones(1, i); err != nil {
				return nil, nil, err
			}
			ni, no := cloneTransitionGraph(i, o)
			m.buf[name] = [2][]*Node{ni, no}
			return i, o, nil
		}
		if err := m.reserve(1); err != nil {
			return nil, nil, err
		}
		n := newNode(v)
		entrypoints = append(entrypoints, n)
		endpoints = append(endpoints, n)
		return

	case ElemOption:
		elemi, elemo, err := m.altGraph(v.Alternation)
		if err != nil {
			return nil, nil, err
		}
		elemi = appendNodes(elemi, emptyNode)
		elemo = appendNodes(elemo, emptyNode)
		return elemi, elemo, nil

	case ElemGroup:
		return m.altGraph(v.Alternation)
	case ElemProseVal:
		return nil, nil, ErrProseValInTransitionGraph
	}

	panic("unsupported element")
}

// newID returns a random identifier for a graph node, the way containerd and
// containers/storage build container IDs: random bytes from crypto/rand, hex
// encoded. Node IDs only have to be unique within one graph, so 16 bytes are
// plenty. crypto/rand.Read never returns an error since Go 1.24.
func newID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func newNode(elem ElemItf) *Node {
	id := newID()
	return &Node{
		ID:   id,
		Elem: elem,
		// Final node, no need to pipe I/O
	}
}

// ToMermaid produces a mermaid-encoded representation of the transition graph.
func (tg *TransitionGraph) ToMermaid() string {
	// Map all nodes
	mp := map[string]*Node{}
	for _, node := range tg.Entrypoints {
		if node == emptyNode {
			continue
		}
		mapNodes(mp, node)
	}

	// Write them all
	str := "flowchart LR\n    classDef entrypoint fill:#479abf\n\n"
	for _, node := range mp {
		// Write down the node
		nodeStr := "    " + node.ID
		elemStr := elemToString(node.Elem)
		if slices.Contains(tg.Endpoints, node) {
			nodeStr += "(((\"" + elemStr + "\")))"
		} else {
			nodeStr += "[\"" + elemStr + "\"]"
		}
		str += nodeStr
		if slices.Contains(tg.Entrypoints, node) {
			str += ":::entrypoint"
		}
		str += "\n"

		// Write down its links
		for _, next := range node.Nexts {
			str += fmt.Sprintf("    %s --> %s\n", node.ID, next.ID)
		}
		str += "\n"
	}

	// Don't forget the empty node
	if slices.Contains(tg.Entrypoints, emptyNode) {
		str += "    emptyNode(((\u2205))):::entrypoint\n"
	}

	return str
}

func mapNodes(mp map[string]*Node, node *Node) {
	if _, ok := mp[node.ID]; ok {
		return
	}
	mp[node.ID] = node
	for _, next := range node.Nexts {
		mapNodes(mp, next)
	}
}

func elemToString(elem ElemItf) string {
	switch v := elem.(type) {
	case ElemRulename:
		return v.Name
	case ElemCharVal:
		str := ""
		for _, b := range v.Values {
			if b == '`' {
				str += "backquote"
			} else if b == '"' {
				str += "dquote"
			} else if 33 <= b && b <= 127 {
				str += string(b)
			}
		}
		return str
	case ElemNumVal:
		if v.Status == StatRange {
			return fmt.Sprintf("0%s%s-%s", v.Base, v.Elems[0], v.Elems[1])
		}
		return fmt.Sprintf("0%s%s", v.Base, strings.Join(v.Elems, "."))
	}
	panic("not implemented yet")
}
