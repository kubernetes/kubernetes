package suffixtree

import (
	"bytes"
	"fmt"
	"math"
	"strings"
)

const infinity = math.MaxInt32

// Pos denotes position in data slice.
type Pos int32

type Token interface {
	Val() int
}

// STree is a struct representing a suffix tree.
type STree struct {
	data     []Token
	root     *state
	auxState *state // auxiliary state

	// active point
	s          *state
	start, end Pos
}

// New creates new suffix tree.
func New() *STree {
	t := new(STree)
	t.data = make([]Token, 0, 50)
	t.root = newState(t)
	t.auxState = newState(t)
	t.root.linkState = t.auxState
	t.s = t.root
	return t
}

// Update refreshes the suffix tree to by new data.
func (t *STree) Update(data ...Token) {
	t.data = append(t.data, data...)
	for _ = range data {
		t.update()
		t.s, t.start = t.canonize(t.s, t.start, t.end)
		t.end++
	}
}

// update transforms suffix tree T(n) to T(n+1).
func (t *STree) update() {
	oldr := t.root

	// (s, (start, end)) is the canonical reference pair for the active point
	s := t.s
	start, end := t.start, t.end
	var r *state
	for {
		var endPoint bool
		r, endPoint = t.testAndSplit(s, start, end-1)
		if endPoint {
			break
		}
		r.fork(end)
		if oldr != t.root {
			oldr.linkState = r
		}
		oldr = r
		s, start = t.canonize(s.linkState, start, end-1)
	}
	if oldr != t.root {
		oldr.linkState = r
	}

	// update active point
	t.s = s
	t.start = start
}

// testAndSplit tests whether a state with canonical ref. pair
// (s, (start, end)) is the end point, that is, a state that have
// a c-transition. If not, then state (exs, (start, end)) is made
// explicit (if not already so).
func (t *STree) testAndSplit(s *state, start, end Pos) (exs *state, endPoint bool) {
	c := t.data[t.end]
	if start <= end {
		tr := s.findTran(t.data[start])
		splitPoint := tr.start + end - start + 1
		if t.data[splitPoint].Val() == c.Val() {
			return s, true
		}
		// make the (s, (start, end)) state explicit
		newSt := newState(s.tree)
		newSt.addTran(splitPoint, tr.end, tr.state)
		tr.end = splitPoint - 1
		tr.state = newSt
		return newSt, false
	}
	if s == t.auxState || s.findTran(c) != nil {
		return s, true
	}
	return s, false
}

// canonize returns updated state and start position for ref. pair
// (s, (start, end)) of state r so the new ref. pair is canonical,
// that is, referenced from the closest explicit ancestor of r.
func (t *STree) canonize(s *state, start, end Pos) (*state, Pos) {
	if s == t.auxState {
		s, start = t.root, start+1
	}
	if start > end {
		return s, start
	}

	var tr *tran
	for {
		if start <= end {
			tr = s.findTran(t.data[start])
			if tr == nil {
				panic(fmt.Sprintf("there should be some transition for '%d' at %d",
					t.data[start].Val(), start))
			}
		}
		if tr.end-tr.start > end-start {
			break
		}
		start += tr.end - tr.start + 1
		s = tr.state
	}
	if s == nil {
		panic("there should always be some suffix link resolution")
	}
	return s, start
}

func (t *STree) At(p Pos) Token {
	if p < 0 || p >= Pos(len(t.data)) {
		panic("position out of bounds")
	}
	return t.data[p]
}

func (t *STree) String() string {
	buf := new(bytes.Buffer)
	printState(buf, t.root, 0)
	return buf.String()
}

func printState(buf *bytes.Buffer, s *state, ident int) {
	for _, tr := range s.trans {
		fmt.Fprint(buf, strings.Repeat("  ", ident))
		fmt.Fprintf(buf, "* (%d, %d)\n", tr.start, tr.ActEnd())
		printState(buf, tr.state, ident+1)
	}
}

// state is an explicit state of the suffix tree.
type state struct {
	tree      *STree
	trans     []*tran
	linkState *state
}

func newState(t *STree) *state {
	return &state{
		tree:      t,
		trans:     make([]*tran, 0),
		linkState: nil,
	}
}

func (s *state) addTran(start, end Pos, r *state) {
	s.trans = append(s.trans, newTran(start, end, r))
}

// fork creates a new branch from the state s.
func (s *state) fork(i Pos) *state {
	r := newState(s.tree)
	s.addTran(i, infinity, r)
	return r
}

// findTran finds c-transition.
func (s *state) findTran(c Token) *tran {
	for _, tran := range s.trans {
		if s.tree.data[tran.start].Val() == c.Val() {
			return tran
		}
	}
	return nil
}

// tran represents a state's transition.
type tran struct {
	start, end Pos
	state      *state
}

func newTran(start, end Pos, s *state) *tran {
	return &tran{start, end, s}
}

func (t *tran) len() int {
	return int(t.end - t.start + 1)
}

// ActEnd returns actual end position as consistent with
// the actual length of the data in the STree.
func (t *tran) ActEnd() Pos {
	if t.end == infinity {
		return Pos(len(t.state.tree.data)) - 1
	}
	return t.end
}
