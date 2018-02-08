// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package adt

import (
	"bytes"
	"math"
)

// Comparable is an interface for trichotomic comparisons.
type Comparable interface {
	// Compare gives the result of a 3-way comparison
	// a.Compare(b) = 1 => a > b
	// a.Compare(b) = 0 => a == b
	// a.Compare(b) = -1 => a < b
	Compare(c Comparable) int
}

type rbcolor int

const (
	black rbcolor = iota
	red
)

// Interval implements a Comparable interval [begin, end)
// TODO: support different sorts of intervals: (a,b), [a,b], (a, b]
type Interval struct {
	Begin Comparable
	End   Comparable
}

// Compare on an interval gives == if the interval overlaps.
func (ivl *Interval) Compare(c Comparable) int {
	ivl2 := c.(*Interval)
	ivbCmpBegin := ivl.Begin.Compare(ivl2.Begin)
	ivbCmpEnd := ivl.Begin.Compare(ivl2.End)
	iveCmpBegin := ivl.End.Compare(ivl2.Begin)

	// ivl is left of ivl2
	if ivbCmpBegin < 0 && iveCmpBegin <= 0 {
		return -1
	}

	// iv is right of iv2
	if ivbCmpEnd >= 0 {
		return 1
	}

	return 0
}

type intervalNode struct {
	// iv is the interval-value pair entry.
	iv IntervalValue
	// max endpoint of all descendent nodes.
	max Comparable
	// left and right are sorted by low endpoint of key interval
	left, right *intervalNode
	// parent is the direct ancestor of the node
	parent *intervalNode
	c      rbcolor
}

func (x *intervalNode) color() rbcolor {
	if x == nil {
		return black
	}
	return x.c
}

func (n *intervalNode) height() int {
	if n == nil {
		return 0
	}
	ld := n.left.height()
	rd := n.right.height()
	if ld < rd {
		return rd + 1
	}
	return ld + 1
}

func (x *intervalNode) min() *intervalNode {
	for x.left != nil {
		x = x.left
	}
	return x
}

// successor is the next in-order node in the tree
func (x *intervalNode) successor() *intervalNode {
	if x.right != nil {
		return x.right.min()
	}
	y := x.parent
	for y != nil && x == y.right {
		x = y
		y = y.parent
	}
	return y
}

// updateMax updates the maximum values for a node and its ancestors
func (x *intervalNode) updateMax() {
	for x != nil {
		oldmax := x.max
		max := x.iv.Ivl.End
		if x.left != nil && x.left.max.Compare(max) > 0 {
			max = x.left.max
		}
		if x.right != nil && x.right.max.Compare(max) > 0 {
			max = x.right.max
		}
		if oldmax.Compare(max) == 0 {
			break
		}
		x.max = max
		x = x.parent
	}
}

type nodeVisitor func(n *intervalNode) bool

// visit will call a node visitor on each node that overlaps the given interval
func (x *intervalNode) visit(iv *Interval, nv nodeVisitor) bool {
	if x == nil {
		return true
	}
	v := iv.Compare(&x.iv.Ivl)
	switch {
	case v < 0:
		if !x.left.visit(iv, nv) {
			return false
		}
	case v > 0:
		maxiv := Interval{x.iv.Ivl.Begin, x.max}
		if maxiv.Compare(iv) == 0 {
			if !x.left.visit(iv, nv) || !x.right.visit(iv, nv) {
				return false
			}
		}
	default:
		if !x.left.visit(iv, nv) || !nv(x) || !x.right.visit(iv, nv) {
			return false
		}
	}
	return true
}

type IntervalValue struct {
	Ivl Interval
	Val interface{}
}

// IntervalTree represents a (mostly) textbook implementation of the
// "Introduction to Algorithms" (Cormen et al, 2nd ed.) chapter 13 red-black tree
// and chapter 14.3 interval tree with search supporting "stabbing queries".
type IntervalTree struct {
	root  *intervalNode
	count int
}

// Delete removes the node with the given interval from the tree, returning
// true if a node is in fact removed.
func (ivt *IntervalTree) Delete(ivl Interval) bool {
	z := ivt.find(ivl)
	if z == nil {
		return false
	}

	y := z
	if z.left != nil && z.right != nil {
		y = z.successor()
	}

	x := y.left
	if x == nil {
		x = y.right
	}
	if x != nil {
		x.parent = y.parent
	}

	if y.parent == nil {
		ivt.root = x
	} else {
		if y == y.parent.left {
			y.parent.left = x
		} else {
			y.parent.right = x
		}
		y.parent.updateMax()
	}
	if y != z {
		z.iv = y.iv
		z.updateMax()
	}

	if y.color() == black && x != nil {
		ivt.deleteFixup(x)
	}

	ivt.count--
	return true
}

func (ivt *IntervalTree) deleteFixup(x *intervalNode) {
	for x != ivt.root && x.color() == black && x.parent != nil {
		if x == x.parent.left {
			w := x.parent.right
			if w.color() == red {
				w.c = black
				x.parent.c = red
				ivt.rotateLeft(x.parent)
				w = x.parent.right
			}
			if w == nil {
				break
			}
			if w.left.color() == black && w.right.color() == black {
				w.c = red
				x = x.parent
			} else {
				if w.right.color() == black {
					w.left.c = black
					w.c = red
					ivt.rotateRight(w)
					w = x.parent.right
				}
				w.c = x.parent.color()
				x.parent.c = black
				w.right.c = black
				ivt.rotateLeft(x.parent)
				x = ivt.root
			}
		} else {
			// same as above but with left and right exchanged
			w := x.parent.left
			if w.color() == red {
				w.c = black
				x.parent.c = red
				ivt.rotateRight(x.parent)
				w = x.parent.left
			}
			if w == nil {
				break
			}
			if w.left.color() == black && w.right.color() == black {
				w.c = red
				x = x.parent
			} else {
				if w.left.color() == black {
					w.right.c = black
					w.c = red
					ivt.rotateLeft(w)
					w = x.parent.left
				}
				w.c = x.parent.color()
				x.parent.c = black
				w.left.c = black
				ivt.rotateRight(x.parent)
				x = ivt.root
			}
		}
	}
	if x != nil {
		x.c = black
	}
}

// Insert adds a node with the given interval into the tree.
func (ivt *IntervalTree) Insert(ivl Interval, val interface{}) {
	var y *intervalNode
	z := &intervalNode{iv: IntervalValue{ivl, val}, max: ivl.End, c: red}
	x := ivt.root
	for x != nil {
		y = x
		if z.iv.Ivl.Begin.Compare(x.iv.Ivl.Begin) < 0 {
			x = x.left
		} else {
			x = x.right
		}
	}

	z.parent = y
	if y == nil {
		ivt.root = z
	} else {
		if z.iv.Ivl.Begin.Compare(y.iv.Ivl.Begin) < 0 {
			y.left = z
		} else {
			y.right = z
		}
		y.updateMax()
	}
	z.c = red
	ivt.insertFixup(z)
	ivt.count++
}

func (ivt *IntervalTree) insertFixup(z *intervalNode) {
	for z.parent != nil && z.parent.parent != nil && z.parent.color() == red {
		if z.parent == z.parent.parent.left {
			y := z.parent.parent.right
			if y.color() == red {
				y.c = black
				z.parent.c = black
				z.parent.parent.c = red
				z = z.parent.parent
			} else {
				if z == z.parent.right {
					z = z.parent
					ivt.rotateLeft(z)
				}
				z.parent.c = black
				z.parent.parent.c = red
				ivt.rotateRight(z.parent.parent)
			}
		} else {
			// same as then with left/right exchanged
			y := z.parent.parent.left
			if y.color() == red {
				y.c = black
				z.parent.c = black
				z.parent.parent.c = red
				z = z.parent.parent
			} else {
				if z == z.parent.left {
					z = z.parent
					ivt.rotateRight(z)
				}
				z.parent.c = black
				z.parent.parent.c = red
				ivt.rotateLeft(z.parent.parent)
			}
		}
	}
	ivt.root.c = black
}

// rotateLeft moves x so it is left of its right child
func (ivt *IntervalTree) rotateLeft(x *intervalNode) {
	y := x.right
	x.right = y.left
	if y.left != nil {
		y.left.parent = x
	}
	x.updateMax()
	ivt.replaceParent(x, y)
	y.left = x
	y.updateMax()
}

// rotateLeft moves x so it is right of its left child
func (ivt *IntervalTree) rotateRight(x *intervalNode) {
	if x == nil {
		return
	}
	y := x.left
	x.left = y.right
	if y.right != nil {
		y.right.parent = x
	}
	x.updateMax()
	ivt.replaceParent(x, y)
	y.right = x
	y.updateMax()
}

// replaceParent replaces x's parent with y
func (ivt *IntervalTree) replaceParent(x *intervalNode, y *intervalNode) {
	y.parent = x.parent
	if x.parent == nil {
		ivt.root = y
	} else {
		if x == x.parent.left {
			x.parent.left = y
		} else {
			x.parent.right = y
		}
		x.parent.updateMax()
	}
	x.parent = y
}

// Len gives the number of elements in the tree
func (ivt *IntervalTree) Len() int { return ivt.count }

// Height is the number of levels in the tree; one node has height 1.
func (ivt *IntervalTree) Height() int { return ivt.root.height() }

// MaxHeight is the expected maximum tree height given the number of nodes
func (ivt *IntervalTree) MaxHeight() int {
	return int((2 * math.Log2(float64(ivt.Len()+1))) + 0.5)
}

// IntervalVisitor is used on tree searches; return false to stop searching.
type IntervalVisitor func(n *IntervalValue) bool

// Visit calls a visitor function on every tree node intersecting the given interval.
// It will visit each interval [x, y) in ascending order sorted on x.
func (ivt *IntervalTree) Visit(ivl Interval, ivv IntervalVisitor) {
	ivt.root.visit(&ivl, func(n *intervalNode) bool { return ivv(&n.iv) })
}

// find the exact node for a given interval
func (ivt *IntervalTree) find(ivl Interval) (ret *intervalNode) {
	f := func(n *intervalNode) bool {
		if n.iv.Ivl != ivl {
			return true
		}
		ret = n
		return false
	}
	ivt.root.visit(&ivl, f)
	return ret
}

// Find gets the IntervalValue for the node matching the given interval
func (ivt *IntervalTree) Find(ivl Interval) (ret *IntervalValue) {
	n := ivt.find(ivl)
	if n == nil {
		return nil
	}
	return &n.iv
}

// Intersects returns true if there is some tree node intersecting the given interval.
func (ivt *IntervalTree) Intersects(iv Interval) bool {
	x := ivt.root
	for x != nil && iv.Compare(&x.iv.Ivl) != 0 {
		if x.left != nil && x.left.max.Compare(iv.Begin) > 0 {
			x = x.left
		} else {
			x = x.right
		}
	}
	return x != nil
}

// Contains returns true if the interval tree's keys cover the entire given interval.
func (ivt *IntervalTree) Contains(ivl Interval) bool {
	var maxEnd, minBegin Comparable

	isContiguous := true
	ivt.Visit(ivl, func(n *IntervalValue) bool {
		if minBegin == nil {
			minBegin = n.Ivl.Begin
			maxEnd = n.Ivl.End
			return true
		}
		if maxEnd.Compare(n.Ivl.Begin) < 0 {
			isContiguous = false
			return false
		}
		if n.Ivl.End.Compare(maxEnd) > 0 {
			maxEnd = n.Ivl.End
		}
		return true
	})

	return isContiguous && minBegin != nil && maxEnd.Compare(ivl.End) >= 0 && minBegin.Compare(ivl.Begin) <= 0
}

// Stab returns a slice with all elements in the tree intersecting the interval.
func (ivt *IntervalTree) Stab(iv Interval) (ivs []*IntervalValue) {
	if ivt.count == 0 {
		return nil
	}
	f := func(n *IntervalValue) bool { ivs = append(ivs, n); return true }
	ivt.Visit(iv, f)
	return ivs
}

type StringComparable string

func (s StringComparable) Compare(c Comparable) int {
	sc := c.(StringComparable)
	if s < sc {
		return -1
	}
	if s > sc {
		return 1
	}
	return 0
}

func NewStringInterval(begin, end string) Interval {
	return Interval{StringComparable(begin), StringComparable(end)}
}

func NewStringPoint(s string) Interval {
	return Interval{StringComparable(s), StringComparable(s + "\x00")}
}

// StringAffineComparable treats "" as > all other strings
type StringAffineComparable string

func (s StringAffineComparable) Compare(c Comparable) int {
	sc := c.(StringAffineComparable)

	if len(s) == 0 {
		if len(sc) == 0 {
			return 0
		}
		return 1
	}
	if len(sc) == 0 {
		return -1
	}

	if s < sc {
		return -1
	}
	if s > sc {
		return 1
	}
	return 0
}

func NewStringAffineInterval(begin, end string) Interval {
	return Interval{StringAffineComparable(begin), StringAffineComparable(end)}
}
func NewStringAffinePoint(s string) Interval {
	return NewStringAffineInterval(s, s+"\x00")
}

func NewInt64Interval(a int64, b int64) Interval {
	return Interval{Int64Comparable(a), Int64Comparable(b)}
}

func NewInt64Point(a int64) Interval {
	return Interval{Int64Comparable(a), Int64Comparable(a + 1)}
}

type Int64Comparable int64

func (v Int64Comparable) Compare(c Comparable) int {
	vc := c.(Int64Comparable)
	cmp := v - vc
	if cmp < 0 {
		return -1
	}
	if cmp > 0 {
		return 1
	}
	return 0
}

// BytesAffineComparable treats empty byte arrays as > all other byte arrays
type BytesAffineComparable []byte

func (b BytesAffineComparable) Compare(c Comparable) int {
	bc := c.(BytesAffineComparable)

	if len(b) == 0 {
		if len(bc) == 0 {
			return 0
		}
		return 1
	}
	if len(bc) == 0 {
		return -1
	}

	return bytes.Compare(b, bc)
}

func NewBytesAffineInterval(begin, end []byte) Interval {
	return Interval{BytesAffineComparable(begin), BytesAffineComparable(end)}
}
func NewBytesAffinePoint(b []byte) Interval {
	be := make([]byte, len(b)+1)
	copy(be, b)
	be[len(b)] = 0
	return NewBytesAffineInterval(b, be)
}
