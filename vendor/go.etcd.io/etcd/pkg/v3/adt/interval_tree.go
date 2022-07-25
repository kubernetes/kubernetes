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
	"fmt"
	"math"
	"strings"
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

func (c rbcolor) String() string {
	switch c {
	case black:
		return "black"
	case red:
		return "red"
	default:
		panic(fmt.Errorf("unknown color %d", c))
	}
}

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

func (x *intervalNode) color(sentinel *intervalNode) rbcolor {
	if x == sentinel {
		return black
	}
	return x.c
}

func (x *intervalNode) height(sentinel *intervalNode) int {
	if x == sentinel {
		return 0
	}
	ld := x.left.height(sentinel)
	rd := x.right.height(sentinel)
	if ld < rd {
		return rd + 1
	}
	return ld + 1
}

func (x *intervalNode) min(sentinel *intervalNode) *intervalNode {
	for x.left != sentinel {
		x = x.left
	}
	return x
}

// successor is the next in-order node in the tree
func (x *intervalNode) successor(sentinel *intervalNode) *intervalNode {
	if x.right != sentinel {
		return x.right.min(sentinel)
	}
	y := x.parent
	for y != sentinel && x == y.right {
		x = y
		y = y.parent
	}
	return y
}

// updateMax updates the maximum values for a node and its ancestors
func (x *intervalNode) updateMax(sentinel *intervalNode) {
	for x != sentinel {
		oldmax := x.max
		max := x.iv.Ivl.End
		if x.left != sentinel && x.left.max.Compare(max) > 0 {
			max = x.left.max
		}
		if x.right != sentinel && x.right.max.Compare(max) > 0 {
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
func (x *intervalNode) visit(iv *Interval, sentinel *intervalNode, nv nodeVisitor) bool {
	if x == sentinel {
		return true
	}
	v := iv.Compare(&x.iv.Ivl)
	switch {
	case v < 0:
		if !x.left.visit(iv, sentinel, nv) {
			return false
		}
	case v > 0:
		maxiv := Interval{x.iv.Ivl.Begin, x.max}
		if maxiv.Compare(iv) == 0 {
			if !x.left.visit(iv, sentinel, nv) || !x.right.visit(iv, sentinel, nv) {
				return false
			}
		}
	default:
		if !x.left.visit(iv, sentinel, nv) || !nv(x) || !x.right.visit(iv, sentinel, nv) {
			return false
		}
	}
	return true
}

// IntervalValue represents a range tree node that contains a range and a value.
type IntervalValue struct {
	Ivl Interval
	Val interface{}
}

// IntervalTree represents a (mostly) textbook implementation of the
// "Introduction to Algorithms" (Cormen et al, 3rd ed.) chapter 13 red-black tree
// and chapter 14.3 interval tree with search supporting "stabbing queries".
type IntervalTree interface {
	// Insert adds a node with the given interval into the tree.
	Insert(ivl Interval, val interface{})
	// Delete removes the node with the given interval from the tree, returning
	// true if a node is in fact removed.
	Delete(ivl Interval) bool
	// Len gives the number of elements in the tree.
	Len() int
	// Height is the number of levels in the tree; one node has height 1.
	Height() int
	// MaxHeight is the expected maximum tree height given the number of nodes.
	MaxHeight() int
	// Visit calls a visitor function on every tree node intersecting the given interval.
	// It will visit each interval [x, y) in ascending order sorted on x.
	Visit(ivl Interval, ivv IntervalVisitor)
	// Find gets the IntervalValue for the node matching the given interval
	Find(ivl Interval) *IntervalValue
	// Intersects returns true if there is some tree node intersecting the given interval.
	Intersects(iv Interval) bool
	// Contains returns true if the interval tree's keys cover the entire given interval.
	Contains(ivl Interval) bool
	// Stab returns a slice with all elements in the tree intersecting the interval.
	Stab(iv Interval) []*IntervalValue
	// Union merges a given interval tree into the receiver.
	Union(inIvt IntervalTree, ivl Interval)
}

// NewIntervalTree returns a new interval tree.
func NewIntervalTree() IntervalTree {
	sentinel := &intervalNode{
		iv:     IntervalValue{},
		max:    nil,
		left:   nil,
		right:  nil,
		parent: nil,
		c:      black,
	}
	return &intervalTree{
		root:     sentinel,
		count:    0,
		sentinel: sentinel,
	}
}

type intervalTree struct {
	root  *intervalNode
	count int

	// red-black NIL node
	// use 'sentinel' as a dummy object to simplify boundary conditions
	// use the sentinel to treat a nil child of a node x as an ordinary node whose parent is x
	// use one shared sentinel to represent all nil leaves and the root's parent
	sentinel *intervalNode
}

// TODO: make this consistent with textbook implementation
//
// "Introduction to Algorithms" (Cormen et al, 3rd ed.), chapter 13.4, p324
//
//	 0. RB-DELETE(T, z)
//	 1.
//	 2. y = z
//	 3. y-original-color = y.color
//	 4.
//	 5. if z.left == T.nil
//	 6. 	x = z.right
//	 7. 	RB-TRANSPLANT(T, z, z.right)
//	 8. else if z.right == T.nil
//	 9. 	x = z.left
//	10. 	RB-TRANSPLANT(T, z, z.left)
//	11. else
//	12. 	y = TREE-MINIMUM(z.right)
//	13. 	y-original-color = y.color
//	14. 	x = y.right
//	15. 	if y.p == z
//	16. 		x.p = y
//	17. 	else
//	18. 		RB-TRANSPLANT(T, y, y.right)
//	19. 		y.right = z.right
//	20. 		y.right.p = y
//	21. 	RB-TRANSPLANT(T, z, y)
//	22. 	y.left = z.left
//	23. 	y.left.p = y
//	24. 	y.color = z.color
//	25.
//	26. if y-original-color == BLACK
//	27. 	RB-DELETE-FIXUP(T, x)

// Delete removes the node with the given interval from the tree, returning
// true if a node is in fact removed.
func (ivt *intervalTree) Delete(ivl Interval) bool {
	z := ivt.find(ivl)
	if z == ivt.sentinel {
		return false
	}

	y := z
	if z.left != ivt.sentinel && z.right != ivt.sentinel {
		y = z.successor(ivt.sentinel)
	}

	x := ivt.sentinel
	if y.left != ivt.sentinel {
		x = y.left
	} else if y.right != ivt.sentinel {
		x = y.right
	}

	x.parent = y.parent

	if y.parent == ivt.sentinel {
		ivt.root = x
	} else {
		if y == y.parent.left {
			y.parent.left = x
		} else {
			y.parent.right = x
		}
		y.parent.updateMax(ivt.sentinel)
	}
	if y != z {
		z.iv = y.iv
		z.updateMax(ivt.sentinel)
	}

	if y.color(ivt.sentinel) == black {
		ivt.deleteFixup(x)
	}

	ivt.count--
	return true
}

// "Introduction to Algorithms" (Cormen et al, 3rd ed.), chapter 13.4, p326
//
//	 0. RB-DELETE-FIXUP(T, z)
//	 1.
//	 2. while x ≠ T.root and x.color == BLACK
//	 3. 	if x == x.p.left
//	 4. 		w = x.p.right
//	 5. 		if w.color == RED
//	 6. 			w.color = BLACK
//	 7. 			x.p.color = RED
//	 8. 			LEFT-ROTATE(T, x, p)
//	 9. 		if w.left.color == BLACK and w.right.color == BLACK
//	10. 			w.color = RED
//	11. 			x = x.p
//	12. 		else if w.right.color == BLACK
//	13. 				w.left.color = BLACK
//	14. 				w.color = RED
//	15. 				RIGHT-ROTATE(T, w)
//	16. 				w = w.p.right
//	17. 			w.color = x.p.color
//	18. 			x.p.color = BLACK
//	19. 			LEFT-ROTATE(T, w.p)
//	20. 			x = T.root
//	21. 	else
//	22. 		w = x.p.left
//	23. 		if w.color == RED
//	24. 			w.color = BLACK
//	25. 			x.p.color = RED
//	26. 			RIGHT-ROTATE(T, x, p)
//	27. 		if w.right.color == BLACK and w.left.color == BLACK
//	28. 			w.color = RED
//	29. 			x = x.p
//	30. 		else if w.left.color == BLACK
//	31. 				w.right.color = BLACK
//	32. 				w.color = RED
//	33. 				LEFT-ROTATE(T, w)
//	34. 				w = w.p.left
//	35. 			w.color = x.p.color
//	36. 			x.p.color = BLACK
//	37. 			RIGHT-ROTATE(T, w.p)
//	38. 			x = T.root
//	39.
//	40. x.color = BLACK
//
func (ivt *intervalTree) deleteFixup(x *intervalNode) {
	for x != ivt.root && x.color(ivt.sentinel) == black {
		if x == x.parent.left { // line 3-20
			w := x.parent.right
			if w.color(ivt.sentinel) == red {
				w.c = black
				x.parent.c = red
				ivt.rotateLeft(x.parent)
				w = x.parent.right
			}
			if w == nil {
				break
			}
			if w.left.color(ivt.sentinel) == black && w.right.color(ivt.sentinel) == black {
				w.c = red
				x = x.parent
			} else {
				if w.right.color(ivt.sentinel) == black {
					w.left.c = black
					w.c = red
					ivt.rotateRight(w)
					w = x.parent.right
				}
				w.c = x.parent.color(ivt.sentinel)
				x.parent.c = black
				w.right.c = black
				ivt.rotateLeft(x.parent)
				x = ivt.root
			}
		} else { // line 22-38
			// same as above but with left and right exchanged
			w := x.parent.left
			if w.color(ivt.sentinel) == red {
				w.c = black
				x.parent.c = red
				ivt.rotateRight(x.parent)
				w = x.parent.left
			}
			if w == nil {
				break
			}
			if w.left.color(ivt.sentinel) == black && w.right.color(ivt.sentinel) == black {
				w.c = red
				x = x.parent
			} else {
				if w.left.color(ivt.sentinel) == black {
					w.right.c = black
					w.c = red
					ivt.rotateLeft(w)
					w = x.parent.left
				}
				w.c = x.parent.color(ivt.sentinel)
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

func (ivt *intervalTree) createIntervalNode(ivl Interval, val interface{}) *intervalNode {
	return &intervalNode{
		iv:     IntervalValue{ivl, val},
		max:    ivl.End,
		c:      red,
		left:   ivt.sentinel,
		right:  ivt.sentinel,
		parent: ivt.sentinel,
	}
}

// TODO: make this consistent with textbook implementation
//
// "Introduction to Algorithms" (Cormen et al, 3rd ed.), chapter 13.3, p315
//
//	 0. RB-INSERT(T, z)
//	 1.
//	 2. y = T.nil
//	 3. x = T.root
//	 4.
//	 5. while x ≠ T.nil
//	 6. 	y = x
//	 7. 	if z.key < x.key
//	 8. 		x = x.left
//	 9. 	else
//	10. 		x = x.right
//	11.
//	12. z.p = y
//	13.
//	14. if y == T.nil
//	15. 	T.root = z
//	16. else if z.key < y.key
//	17. 	y.left = z
//	18. else
//	19. 	y.right = z
//	20.
//	21. z.left = T.nil
//	22. z.right = T.nil
//	23. z.color = RED
//	24.
//	25. RB-INSERT-FIXUP(T, z)

// Insert adds a node with the given interval into the tree.
func (ivt *intervalTree) Insert(ivl Interval, val interface{}) {
	y := ivt.sentinel
	z := ivt.createIntervalNode(ivl, val)
	x := ivt.root
	for x != ivt.sentinel {
		y = x
		if z.iv.Ivl.Begin.Compare(x.iv.Ivl.Begin) < 0 {
			x = x.left
		} else {
			x = x.right
		}
	}

	z.parent = y
	if y == ivt.sentinel {
		ivt.root = z
	} else {
		if z.iv.Ivl.Begin.Compare(y.iv.Ivl.Begin) < 0 {
			y.left = z
		} else {
			y.right = z
		}
		y.updateMax(ivt.sentinel)
	}
	z.c = red

	ivt.insertFixup(z)
	ivt.count++
}

// "Introduction to Algorithms" (Cormen et al, 3rd ed.), chapter 13.3, p316
//
//	 0. RB-INSERT-FIXUP(T, z)
//	 1.
//	 2. while z.p.color == RED
//	 3. 	if z.p == z.p.p.left
//	 4. 		y = z.p.p.right
//	 5. 		if y.color == RED
//	 6. 			z.p.color = BLACK
//	 7. 			y.color = BLACK
//	 8. 			z.p.p.color = RED
//	 9. 			z = z.p.p
//	10. 		else if z == z.p.right
//	11. 				z = z.p
//	12. 				LEFT-ROTATE(T, z)
//	13. 			z.p.color = BLACK
//	14. 			z.p.p.color = RED
//	15. 			RIGHT-ROTATE(T, z.p.p)
//	16. 	else
//	17. 		y = z.p.p.left
//	18. 		if y.color == RED
//	19. 			z.p.color = BLACK
//	20. 			y.color = BLACK
//	21. 			z.p.p.color = RED
//	22. 			z = z.p.p
//	23. 		else if z == z.p.right
//	24. 				z = z.p
//	25. 				RIGHT-ROTATE(T, z)
//	26. 			z.p.color = BLACK
//	27. 			z.p.p.color = RED
//	28. 			LEFT-ROTATE(T, z.p.p)
//	29.
//	30. T.root.color = BLACK
//
func (ivt *intervalTree) insertFixup(z *intervalNode) {
	for z.parent.color(ivt.sentinel) == red {
		if z.parent == z.parent.parent.left { // line 3-15

			y := z.parent.parent.right
			if y.color(ivt.sentinel) == red {
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
		} else { // line 16-28
			// same as then with left/right exchanged
			y := z.parent.parent.left
			if y.color(ivt.sentinel) == red {
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

	// line 30
	ivt.root.c = black
}

// rotateLeft moves x so it is left of its right child
//
// "Introduction to Algorithms" (Cormen et al, 3rd ed.), chapter 13.2, p313
//
//	 0. LEFT-ROTATE(T, x)
//	 1.
//	 2. y = x.right
//	 3. x.right = y.left
//	 4.
//	 5. if y.left ≠ T.nil
//	 6. 	y.left.p = x
//	 7.
//	 8. y.p = x.p
//	 9.
//	10. if x.p == T.nil
//	11. 	T.root = y
//	12. else if x == x.p.left
//	13. 	x.p.left = y
//	14. else
//	15. 	x.p.right = y
//	16.
//	17. y.left = x
//	18. x.p = y
//
func (ivt *intervalTree) rotateLeft(x *intervalNode) {
	// rotateLeft x must have right child
	if x.right == ivt.sentinel {
		return
	}

	// line 2-3
	y := x.right
	x.right = y.left

	// line 5-6
	if y.left != ivt.sentinel {
		y.left.parent = x
	}
	x.updateMax(ivt.sentinel)

	// line 10-15, 18
	ivt.replaceParent(x, y)

	// line 17
	y.left = x
	y.updateMax(ivt.sentinel)
}

// rotateRight moves x so it is right of its left child
//
//	 0. RIGHT-ROTATE(T, x)
//	 1.
//	 2. y = x.left
//	 3. x.left = y.right
//	 4.
//	 5. if y.right ≠ T.nil
//	 6. 	y.right.p = x
//	 7.
//	 8. y.p = x.p
//	 9.
//	10. if x.p == T.nil
//	11. 	T.root = y
//	12. else if x == x.p.right
//	13. 	x.p.right = y
//	14. else
//	15. 	x.p.left = y
//	16.
//	17. y.right = x
//	18. x.p = y
//
func (ivt *intervalTree) rotateRight(x *intervalNode) {
	// rotateRight x must have left child
	if x.left == ivt.sentinel {
		return
	}

	// line 2-3
	y := x.left
	x.left = y.right

	// line 5-6
	if y.right != ivt.sentinel {
		y.right.parent = x
	}
	x.updateMax(ivt.sentinel)

	// line 10-15, 18
	ivt.replaceParent(x, y)

	// line 17
	y.right = x
	y.updateMax(ivt.sentinel)
}

// replaceParent replaces x's parent with y
func (ivt *intervalTree) replaceParent(x *intervalNode, y *intervalNode) {
	y.parent = x.parent
	if x.parent == ivt.sentinel {
		ivt.root = y
	} else {
		if x == x.parent.left {
			x.parent.left = y
		} else {
			x.parent.right = y
		}
		x.parent.updateMax(ivt.sentinel)
	}
	x.parent = y
}

// Len gives the number of elements in the tree
func (ivt *intervalTree) Len() int { return ivt.count }

// Height is the number of levels in the tree; one node has height 1.
func (ivt *intervalTree) Height() int { return ivt.root.height(ivt.sentinel) }

// MaxHeight is the expected maximum tree height given the number of nodes
func (ivt *intervalTree) MaxHeight() int {
	return int((2 * math.Log2(float64(ivt.Len()+1))) + 0.5)
}

// IntervalVisitor is used on tree searches; return false to stop searching.
type IntervalVisitor func(n *IntervalValue) bool

// Visit calls a visitor function on every tree node intersecting the given interval.
// It will visit each interval [x, y) in ascending order sorted on x.
func (ivt *intervalTree) Visit(ivl Interval, ivv IntervalVisitor) {
	ivt.root.visit(&ivl, ivt.sentinel, func(n *intervalNode) bool { return ivv(&n.iv) })
}

// find the exact node for a given interval
func (ivt *intervalTree) find(ivl Interval) *intervalNode {
	ret := ivt.sentinel
	f := func(n *intervalNode) bool {
		if n.iv.Ivl != ivl {
			return true
		}
		ret = n
		return false
	}
	ivt.root.visit(&ivl, ivt.sentinel, f)
	return ret
}

// Find gets the IntervalValue for the node matching the given interval
func (ivt *intervalTree) Find(ivl Interval) (ret *IntervalValue) {
	n := ivt.find(ivl)
	if n == ivt.sentinel {
		return nil
	}
	return &n.iv
}

// Intersects returns true if there is some tree node intersecting the given interval.
func (ivt *intervalTree) Intersects(iv Interval) bool {
	x := ivt.root
	for x != ivt.sentinel && iv.Compare(&x.iv.Ivl) != 0 {
		if x.left != ivt.sentinel && x.left.max.Compare(iv.Begin) > 0 {
			x = x.left
		} else {
			x = x.right
		}
	}
	return x != ivt.sentinel
}

// Contains returns true if the interval tree's keys cover the entire given interval.
func (ivt *intervalTree) Contains(ivl Interval) bool {
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
func (ivt *intervalTree) Stab(iv Interval) (ivs []*IntervalValue) {
	if ivt.count == 0 {
		return nil
	}
	f := func(n *IntervalValue) bool { ivs = append(ivs, n); return true }
	ivt.Visit(iv, f)
	return ivs
}

// Union merges a given interval tree into the receiver.
func (ivt *intervalTree) Union(inIvt IntervalTree, ivl Interval) {
	f := func(n *IntervalValue) bool {
		ivt.Insert(n.Ivl, n.Val)
		return true
	}
	inIvt.Visit(ivl, f)
}

type visitedInterval struct {
	root  Interval
	left  Interval
	right Interval
	color rbcolor
	depth int
}

func (vi visitedInterval) String() string {
	bd := new(strings.Builder)
	bd.WriteString(fmt.Sprintf("root [%v,%v,%v], left [%v,%v], right [%v,%v], depth %d",
		vi.root.Begin, vi.root.End, vi.color,
		vi.left.Begin, vi.left.End,
		vi.right.Begin, vi.right.End,
		vi.depth,
	))
	return bd.String()
}

// visitLevel traverses tree in level order.
// used for testing
func (ivt *intervalTree) visitLevel() []visitedInterval {
	if ivt.root == ivt.sentinel {
		return nil
	}

	rs := make([]visitedInterval, 0, ivt.Len())

	type pair struct {
		node  *intervalNode
		depth int
	}
	queue := []pair{{ivt.root, 0}}
	for len(queue) > 0 {
		f := queue[0]
		queue = queue[1:]

		vi := visitedInterval{
			root:  f.node.iv.Ivl,
			color: f.node.color(ivt.sentinel),
			depth: f.depth,
		}
		if f.node.left != ivt.sentinel {
			vi.left = f.node.left.iv.Ivl
			queue = append(queue, pair{f.node.left, f.depth + 1})
		}
		if f.node.right != ivt.sentinel {
			vi.right = f.node.right.iv.Ivl
			queue = append(queue, pair{f.node.right, f.depth + 1})
		}

		rs = append(rs, vi)
	}

	return rs
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

func newInt64EmptyInterval() Interval {
	return Interval{Begin: nil, End: nil}
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
