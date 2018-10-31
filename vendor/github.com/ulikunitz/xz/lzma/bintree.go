// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"unicode"
)

// node represents a node in the binary tree.
type node struct {
	// x is the search value
	x uint32
	// p parent node
	p uint32
	// l left child
	l uint32
	// r right child
	r uint32
}

// wordLen is the number of bytes represented by the v field of a node.
const wordLen = 4

// binTree supports the identification of the next operation based on a
// binary tree.
//
// Nodes will be identified by their index into the ring buffer.
type binTree struct {
	dict *encoderDict
	// ring buffer of nodes
	node []node
	// absolute offset of the entry for the next node. Position 4
	// byte larger.
	hoff int64
	// front position in the node ring buffer
	front uint32
	// index of the root node
	root uint32
	// current x value
	x uint32
	// preallocated array
	data []byte
}

// null represents the nonexistent index. We can't use zero because it
// would always exist or we would need to decrease the index for each
// reference.
const null uint32 = 1<<32 - 1

// newBinTree initializes the binTree structure. The capacity defines
// the size of the buffer and defines the maximum distance for which
// matches will be found.
func newBinTree(capacity int) (t *binTree, err error) {
	if capacity < 1 {
		return nil, errors.New(
			"newBinTree: capacity must be larger than zero")
	}
	if int64(capacity) >= int64(null) {
		return nil, errors.New(
			"newBinTree: capacity must less 2^{32}-1")
	}
	t = &binTree{
		node: make([]node, capacity),
		hoff: -int64(wordLen),
		root: null,
		data: make([]byte, maxMatchLen),
	}
	return t, nil
}

func (t *binTree) SetDict(d *encoderDict) { t.dict = d }

// WriteByte writes a single byte into the binary tree.
func (t *binTree) WriteByte(c byte) error {
	t.x = (t.x << 8) | uint32(c)
	t.hoff++
	if t.hoff < 0 {
		return nil
	}
	v := t.front
	if int64(v) < t.hoff {
		// We are overwriting old nodes stored in the tree.
		t.remove(v)
	}
	t.node[v].x = t.x
	t.add(v)
	t.front++
	if int64(t.front) >= int64(len(t.node)) {
		t.front = 0
	}
	return nil
}

// Writes writes a sequence of bytes into the binTree structure.
func (t *binTree) Write(p []byte) (n int, err error) {
	for _, c := range p {
		t.WriteByte(c)
	}
	return len(p), nil
}

// add puts the node v into the tree. The node must not be part of the
// tree before.
func (t *binTree) add(v uint32) {
	vn := &t.node[v]
	// Set left and right to null indices.
	vn.l, vn.r = null, null
	// If the binary tree is empty make v the root.
	if t.root == null {
		t.root = v
		vn.p = null
		return
	}
	x := vn.x
	p := t.root
	// Search for the right leave link and add the new node.
	for {
		pn := &t.node[p]
		if x <= pn.x {
			if pn.l == null {
				pn.l = v
				vn.p = p
				return
			}
			p = pn.l
		} else {
			if pn.r == null {
				pn.r = v
				vn.p = p
				return
			}
			p = pn.r
		}
	}
}

// parent returns the parent node index of v and the pointer to v value
// in the parent.
func (t *binTree) parent(v uint32) (p uint32, ptr *uint32) {
	if t.root == v {
		return null, &t.root
	}
	p = t.node[v].p
	if t.node[p].l == v {
		ptr = &t.node[p].l
	} else {
		ptr = &t.node[p].r
	}
	return
}

// Remove node v.
func (t *binTree) remove(v uint32) {
	vn := &t.node[v]
	p, ptr := t.parent(v)
	l, r := vn.l, vn.r
	if l == null {
		// Move the right child up.
		*ptr = r
		if r != null {
			t.node[r].p = p
		}
		return
	}
	if r == null {
		// Move the left child up.
		*ptr = l
		t.node[l].p = p
		return
	}

	// Search the in-order predecessor u.
	un := &t.node[l]
	ur := un.r
	if ur == null {
		// In order predecessor is l. Move it up.
		un.r = r
		t.node[r].p = l
		un.p = p
		*ptr = l
		return
	}
	var u uint32
	for {
		// Look for the max value in the tree where l is root.
		u = ur
		ur = t.node[u].r
		if ur == null {
			break
		}
	}
	// replace u with ul
	un = &t.node[u]
	ul := un.l
	up := un.p
	t.node[up].r = ul
	if ul != null {
		t.node[ul].p = up
	}

	// replace v by u
	un.l, un.r = l, r
	t.node[l].p = u
	t.node[r].p = u
	*ptr = u
	un.p = p
}

// search looks for the node that have the value x or for the nodes that
// brace it. The node highest in the tree with the value x will be
// returned. All other nodes with the same value live in left subtree of
// the returned node.
func (t *binTree) search(v uint32, x uint32) (a, b uint32) {
	a, b = null, null
	if v == null {
		return
	}
	for {
		vn := &t.node[v]
		if x <= vn.x {
			if x == vn.x {
				return v, v
			}
			b = v
			if vn.l == null {
				return
			}
			v = vn.l
		} else {
			a = v
			if vn.r == null {
				return
			}
			v = vn.r
		}
	}
}

// max returns the node with maximum value in the subtree with v as
// root.
func (t *binTree) max(v uint32) uint32 {
	if v == null {
		return null
	}
	for {
		r := t.node[v].r
		if r == null {
			return v
		}
		v = r
	}
}

// min returns the node with the minimum value in the subtree with v as
// root.
func (t *binTree) min(v uint32) uint32 {
	if v == null {
		return null
	}
	for {
		l := t.node[v].l
		if l == null {
			return v
		}
		v = l
	}
}

// pred returns the in-order predecessor of node v.
func (t *binTree) pred(v uint32) uint32 {
	if v == null {
		return null
	}
	u := t.max(t.node[v].l)
	if u != null {
		return u
	}
	for {
		p := t.node[v].p
		if p == null {
			return null
		}
		if t.node[p].r == v {
			return p
		}
		v = p
	}
}

// succ returns the in-order successor of node v.
func (t *binTree) succ(v uint32) uint32 {
	if v == null {
		return null
	}
	u := t.min(t.node[v].r)
	if u != null {
		return u
	}
	for {
		p := t.node[v].p
		if p == null {
			return null
		}
		if t.node[p].l == v {
			return p
		}
		v = p
	}
}

// xval converts the first four bytes of a into an 32-bit unsigned
// integer in big-endian order.
func xval(a []byte) uint32 {
	var x uint32
	switch len(a) {
	default:
		x |= uint32(a[3])
		fallthrough
	case 3:
		x |= uint32(a[2]) << 8
		fallthrough
	case 2:
		x |= uint32(a[1]) << 16
		fallthrough
	case 1:
		x |= uint32(a[0]) << 24
	case 0:
	}
	return x
}

// dumpX converts value x into a four-letter string.
func dumpX(x uint32) string {
	a := make([]byte, 4)
	for i := 0; i < 4; i++ {
		c := byte(x >> uint((3-i)*8))
		if unicode.IsGraphic(rune(c)) {
			a[i] = c
		} else {
			a[i] = '.'
		}
	}
	return string(a)
}

// dumpNode writes a representation of the node v into the io.Writer.
func (t *binTree) dumpNode(w io.Writer, v uint32, indent int) {
	if v == null {
		return
	}

	vn := &t.node[v]

	t.dumpNode(w, vn.r, indent+2)

	for i := 0; i < indent; i++ {
		fmt.Fprint(w, " ")
	}
	if vn.p == null {
		fmt.Fprintf(w, "node %d %q parent null\n", v, dumpX(vn.x))
	} else {
		fmt.Fprintf(w, "node %d %q parent %d\n", v, dumpX(vn.x), vn.p)
	}

	t.dumpNode(w, vn.l, indent+2)
}

// dump prints a representation of the binary tree into the writer.
func (t *binTree) dump(w io.Writer) error {
	bw := bufio.NewWriter(w)
	t.dumpNode(bw, t.root, 0)
	return bw.Flush()
}

func (t *binTree) distance(v uint32) int {
	dist := int(t.front) - int(v)
	if dist <= 0 {
		dist += len(t.node)
	}
	return dist
}

type matchParams struct {
	rep [4]uint32
	// length when match will be accepted
	nAccept int
	// nodes to check
	check int
	// finish if length get shorter
	stopShorter bool
}

func (t *binTree) match(m match, distIter func() (int, bool), p matchParams,
) (r match, checked int, accepted bool) {
	buf := &t.dict.buf
	for {
		if checked >= p.check {
			return m, checked, true
		}
		dist, ok := distIter()
		if !ok {
			return m, checked, false
		}
		checked++
		if m.n > 0 {
			i := buf.rear - dist + m.n - 1
			if i < 0 {
				i += len(buf.data)
			} else if i >= len(buf.data) {
				i -= len(buf.data)
			}
			if buf.data[i] != t.data[m.n-1] {
				if p.stopShorter {
					return m, checked, false
				}
				continue
			}
		}
		n := buf.matchLen(dist, t.data)
		switch n {
		case 0:
			if p.stopShorter {
				return m, checked, false
			}
			continue
		case 1:
			if uint32(dist-minDistance) != p.rep[0] {
				continue
			}
		}
		if n < m.n || (n == m.n && int64(dist) >= m.distance) {
			continue
		}
		m = match{int64(dist), n}
		if n >= p.nAccept {
			return m, checked, true
		}
	}
}

func (t *binTree) NextOp(rep [4]uint32) operation {
	// retrieve maxMatchLen data
	n, _ := t.dict.buf.Peek(t.data[:maxMatchLen])
	if n == 0 {
		panic("no data in buffer")
	}
	t.data = t.data[:n]

	var (
		m                  match
		x, u, v            uint32
		iterPred, iterSucc func() (int, bool)
	)
	p := matchParams{
		rep:     rep,
		nAccept: maxMatchLen,
		check:   32,
	}
	i := 4
	iterSmall := func() (dist int, ok bool) {
		i--
		if i <= 0 {
			return 0, false
		}
		return i, true
	}
	m, checked, accepted := t.match(m, iterSmall, p)
	if accepted {
		goto end
	}
	p.check -= checked
	x = xval(t.data)
	u, v = t.search(t.root, x)
	if u == v && len(t.data) == 4 {
		iter := func() (dist int, ok bool) {
			if u == null {
				return 0, false
			}
			dist = t.distance(u)
			u, v = t.search(t.node[u].l, x)
			if u != v {
				u = null
			}
			return dist, true
		}
		m, _, _ = t.match(m, iter, p)
		goto end
	}
	p.stopShorter = true
	iterSucc = func() (dist int, ok bool) {
		if v == null {
			return 0, false
		}
		dist = t.distance(v)
		v = t.succ(v)
		return dist, true
	}
	m, checked, accepted = t.match(m, iterSucc, p)
	if accepted {
		goto end
	}
	p.check -= checked
	iterPred = func() (dist int, ok bool) {
		if u == null {
			return 0, false
		}
		dist = t.distance(u)
		u = t.pred(u)
		return dist, true
	}
	m, _, _ = t.match(m, iterPred, p)
end:
	if m.n == 0 {
		return lit{t.data[0]}
	}
	return m
}
