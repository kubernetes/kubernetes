// Copyright 2014 Google LLC
// Modified 2018 by Jonathan Amsterdam (jbamsterdam@gmail.com)
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

// Package btree implements in-memory B-Trees of arbitrary degree.
//
// This implementation is based on google/btree (http://github.com/google/btree), and
// much of the code is taken from there. But the API has been changed significantly,
// particularly around iteration, and support for indexing by position has been
// added.
//
// btree implements an in-memory B-Tree for use as an ordered data structure.
// It is not meant for persistent storage solutions.
//
// It has a flatter structure than an equivalent red-black or other binary tree,
// which in some cases yields better memory usage and/or performance.
// See some discussion on the matter here:
//   http://google-opensource.blogspot.com/2013/01/c-containers-that-save-memory-and-time.html
// Note, though, that this project is in no way related to the C++ B-Tree
// implementation written about there.
//
// Within this tree, each node contains a slice of items and a (possibly nil)
// slice of children.  For basic numeric values or raw structs, this can cause
// efficiency differences when compared to equivalent C++ template code that
// stores values in arrays within the node:
//   * Due to the overhead of storing values as interfaces (each
//     value needs to be stored as the value itself, then 2 words for the
//     interface pointing to that value and its type), resulting in higher
//     memory use.
//   * Since interfaces can point to values anywhere in memory, values are
//     most likely not stored in contiguous blocks, resulting in a higher
//     number of cache misses.
// These issues don't tend to matter, though, when working with strings or other
// heap-allocated structures, since C++-equivalent structures also must store
// pointers and also distribute their values across the heap.
package btree

import (
	"sort"
	"sync"
)

// Key represents a key into the tree.
type Key interface{}

// Value represents a value in the tree.
type Value interface{}

// item is a key-value pair.
type item struct {
	key   Key
	value Value
}

type lessFunc func(interface{}, interface{}) bool

// New creates a new B-Tree with the given degree and comparison function.
//
// New(2, less), for example, will create a 2-3-4 tree (each node contains 1-3 items
// and 2-4 children).
//
// The less function tests whether the current item is less than the given argument.
// It must provide a strict weak ordering.
// If !less(a, b) && !less(b, a), we treat this to mean a == b (i.e. the tree
// can hold only one of a or b).
func New(degree int, less func(interface{}, interface{}) bool) *BTree {
	if degree <= 1 {
		panic("bad degree")
	}
	return &BTree{
		degree: degree,
		less:   less,
		cow:    &copyOnWriteContext{},
	}
}

// items stores items in a node.
type items []item

// insertAt inserts a value into the given index, pushing all subsequent values
// forward.
func (s *items) insertAt(index int, m item) {
	*s = append(*s, item{})
	if index < len(*s) {
		copy((*s)[index+1:], (*s)[index:])
	}
	(*s)[index] = m
}

// removeAt removes a value at a given index, pulling all subsequent values
// back.
func (s *items) removeAt(index int) item {
	m := (*s)[index]
	copy((*s)[index:], (*s)[index+1:])
	(*s)[len(*s)-1] = item{}
	*s = (*s)[:len(*s)-1]
	return m
}

// pop removes and returns the last element in the list.
func (s *items) pop() item {
	index := len(*s) - 1
	out := (*s)[index]
	(*s)[index] = item{}
	*s = (*s)[:index]
	return out
}

var nilItems = make(items, 16)

// truncate truncates this instance at index so that it contains only the
// first index items. index must be less than or equal to length.
func (s *items) truncate(index int) {
	var toClear items
	*s, toClear = (*s)[:index], (*s)[index:]
	for len(toClear) > 0 {
		toClear = toClear[copy(toClear, nilItems):]
	}
}

// find returns the index where an item with key should be inserted into this
// list.  'found' is true if the item already exists in the list at the given
// index.
func (s items) find(k Key, less lessFunc) (index int, found bool) {
	i := sort.Search(len(s), func(i int) bool { return less(k, s[i].key) })
	// i is the smallest index of s for which k.Less(s[i].Key), or len(s).
	if i > 0 && !less(s[i-1].key, k) {
		return i - 1, true
	}
	return i, false
}

// children stores child nodes in a node.
type children []*node

// insertAt inserts a value into the given index, pushing all subsequent values
// forward.
func (s *children) insertAt(index int, n *node) {
	*s = append(*s, nil)
	if index < len(*s) {
		copy((*s)[index+1:], (*s)[index:])
	}
	(*s)[index] = n
}

// removeAt removes a value at a given index, pulling all subsequent values
// back.
func (s *children) removeAt(index int) *node {
	n := (*s)[index]
	copy((*s)[index:], (*s)[index+1:])
	(*s)[len(*s)-1] = nil
	*s = (*s)[:len(*s)-1]
	return n
}

// pop removes and returns the last element in the list.
func (s *children) pop() (out *node) {
	index := len(*s) - 1
	out = (*s)[index]
	(*s)[index] = nil
	*s = (*s)[:index]
	return
}

var nilChildren = make(children, 16)

// truncate truncates this instance at index so that it contains only the
// first index children. index must be less than or equal to length.
func (s *children) truncate(index int) {
	var toClear children
	*s, toClear = (*s)[:index], (*s)[index:]
	for len(toClear) > 0 {
		toClear = toClear[copy(toClear, nilChildren):]
	}
}

// node is an internal node in a tree.
//
// It must at all times maintain the invariant that either
//   * len(children) == 0, len(items) unconstrained
//   * len(children) == len(items) + 1
type node struct {
	items    items
	children children
	size     int // number of items in the subtree: len(items) + sum over i of children[i].size
	cow      *copyOnWriteContext
}

func (n *node) computeSize() int {
	sz := len(n.items)
	for _, c := range n.children {
		sz += c.size
	}
	return sz
}

func (n *node) mutableFor(cow *copyOnWriteContext) *node {
	if n.cow == cow {
		return n
	}
	out := cow.newNode()
	if cap(out.items) >= len(n.items) {
		out.items = out.items[:len(n.items)]
	} else {
		out.items = make(items, len(n.items), cap(n.items))
	}
	copy(out.items, n.items)
	// Copy children
	if cap(out.children) >= len(n.children) {
		out.children = out.children[:len(n.children)]
	} else {
		out.children = make(children, len(n.children), cap(n.children))
	}
	copy(out.children, n.children)
	out.size = n.size
	return out
}

func (n *node) mutableChild(i int) *node {
	c := n.children[i].mutableFor(n.cow)
	n.children[i] = c
	return c
}

// split splits the given node at the given index.  The current node shrinks,
// and this function returns the item that existed at that index and a new node
// containing all items/children after it.
func (n *node) split(i int) (item, *node) {
	item := n.items[i]
	next := n.cow.newNode()
	next.items = append(next.items, n.items[i+1:]...)
	n.items.truncate(i)
	if len(n.children) > 0 {
		next.children = append(next.children, n.children[i+1:]...)
		n.children.truncate(i + 1)
	}
	n.size = n.computeSize()
	next.size = next.computeSize()
	return item, next
}

// maybeSplitChild checks if a child should be split, and if so splits it.
// Returns whether or not a split occurred.
func (n *node) maybeSplitChild(i, maxItems int) bool {
	if len(n.children[i].items) < maxItems {
		return false
	}
	first := n.mutableChild(i)
	item, second := first.split(maxItems / 2)
	n.items.insertAt(i, item)
	n.children.insertAt(i+1, second)
	// The size of n doesn't change.
	return true
}

// insert inserts an item into the subtree rooted at this node, making sure
// no nodes in the subtree exceed maxItems items.  Should an equivalent item be
// be found/replaced by insert, its value will be returned.
//
// If computeIndex is true, the third return value is the index of the value with respect to n.
func (n *node) insert(m item, maxItems int, less lessFunc, computeIndex bool) (old Value, present bool, idx int) {
	i, found := n.items.find(m.key, less)
	if found {
		out := n.items[i]
		n.items[i] = m
		if computeIndex {
			idx = n.itemIndex(i)
		}
		return out.value, true, idx
	}
	if len(n.children) == 0 {
		n.items.insertAt(i, m)
		n.size++
		return old, false, i
	}
	if n.maybeSplitChild(i, maxItems) {
		inTree := n.items[i]
		switch {
		case less(m.key, inTree.key):
			// no change, we want first split node
		case less(inTree.key, m.key):
			i++ // we want second split node
		default:
			out := n.items[i]
			n.items[i] = m
			if computeIndex {
				idx = n.itemIndex(i)
			}
			return out.value, true, idx
		}
	}
	old, present, idx = n.mutableChild(i).insert(m, maxItems, less, computeIndex)
	if !present {
		n.size++
	}
	if computeIndex {
		idx += n.partialSize(i)
	}
	return old, present, idx
}

// get finds the given key in the subtree and returns the corresponding item, along with a boolean reporting
// whether it was found.
// If computeIndex is true, it also returns the index of the key relative to the node's subtree.
func (n *node) get(k Key, computeIndex bool, less lessFunc) (item, bool, int) {
	i, found := n.items.find(k, less)
	if found {
		return n.items[i], true, n.itemIndex(i)
	}
	if len(n.children) > 0 {
		m, found, idx := n.children[i].get(k, computeIndex, less)
		if computeIndex && found {
			idx += n.partialSize(i)
		}
		return m, found, idx
	}
	return item{}, false, -1
}

// itemIndex returns the index w.r.t. n of the ith item in n.
func (n *node) itemIndex(i int) int {
	if len(n.children) == 0 {
		return i
	}
	// Get the size of the node up to but not including the child to the right of
	// item i. Subtract 1 because the index is 0-based.
	return n.partialSize(i+1) - 1
}

// Returns the size of the non-leaf node up to but not including child i.
func (n *node) partialSize(i int) int {
	var sz int
	for j, c := range n.children {
		if j == i {
			break
		}
		sz += c.size + 1
	}
	return sz
}

// cursorStackForKey returns a stack of cursors for the key, along with whether the key was found and the index.
func (n *node) cursorStackForKey(k Key, cs cursorStack, less lessFunc) (cursorStack, bool, int) {
	i, found := n.items.find(k, less)
	cs.push(cursor{n, i})
	idx := i
	if found {
		if len(n.children) > 0 {
			idx = n.partialSize(i+1) - 1
		}
		return cs, true, idx
	}
	if len(n.children) > 0 {
		cs, found, idx := n.children[i].cursorStackForKey(k, cs, less)
		return cs, found, idx + n.partialSize(i)
	}
	return cs, false, idx
}

// at returns the item at the i'th position in the subtree rooted at n.
// It assumes i is in range.
func (n *node) at(i int) item {
	if len(n.children) == 0 {
		return n.items[i]
	}
	for j, c := range n.children {
		if i < c.size {
			return c.at(i)
		}
		i -= c.size
		if i == 0 {
			return n.items[j]
		}
		i--
	}
	panic("impossible")
}

// cursorStackForIndex returns a stack of cursors for the index.
// It assumes i is in range.
func (n *node) cursorStackForIndex(i int, cs cursorStack) cursorStack {
	if len(n.children) == 0 {
		return cs.push(cursor{n, i})
	}
	for j, c := range n.children {
		if i < c.size {
			return c.cursorStackForIndex(i, cs.push(cursor{n, j}))
		}
		i -= c.size
		if i == 0 {
			return cs.push(cursor{n, j})
		}
		i--
	}
	panic("impossible")
}

// toRemove details what item to remove in a node.remove call.
type toRemove int

const (
	removeItem toRemove = iota // removes the given item
	removeMin                  // removes smallest item in the subtree
	removeMax                  // removes largest item in the subtree
)

// remove removes an item from the subtree rooted at this node.
func (n *node) remove(key Key, minItems int, typ toRemove, less lessFunc) (item, bool) {
	var i int
	var found bool
	switch typ {
	case removeMax:
		if len(n.children) == 0 {
			n.size--
			return n.items.pop(), true

		}
		i = len(n.items)
	case removeMin:
		if len(n.children) == 0 {
			n.size--
			return n.items.removeAt(0), true
		}
		i = 0
	case removeItem:
		i, found = n.items.find(key, less)
		if len(n.children) == 0 {
			if found {
				n.size--
				return n.items.removeAt(i), true
			}
			return item{}, false
		}
	default:
		panic("invalid type")
	}
	// If we get to here, we have children.
	if len(n.children[i].items) <= minItems {
		return n.growChildAndRemove(i, key, minItems, typ, less)
	}
	child := n.mutableChild(i)
	// Either we had enough items to begin with, or we've done some
	// merging/stealing, because we've got enough now and we're ready to return
	// stuff.
	if found {
		// The item exists at index 'i', and the child we've selected can give us a
		// predecessor, since if we've gotten here it's got > minItems items in it.
		out := n.items[i]
		// We use our special-case 'remove' call with typ=maxItem to pull the
		// predecessor of item i (the rightmost leaf of our immediate left child)
		// and set it into where we pulled the item from.
		n.items[i], _ = child.remove(nil, minItems, removeMax, less)
		n.size--
		return out, true
	}
	// Final recursive call.  Once we're here, we know that the item isn't in this
	// node and that the child is big enough to remove from.
	m, removed := child.remove(key, minItems, typ, less)
	if removed {
		n.size--
	}
	return m, removed
}

// growChildAndRemove grows child 'i' to make sure it's possible to remove an
// item from it while keeping it at minItems, then calls remove to actually
// remove it.
//
// Most documentation says we have to do two sets of special casing:
//   1) item is in this node
//   2) item is in child
// In both cases, we need to handle the two subcases:
//   A) node has enough values that it can spare one
//   B) node doesn't have enough values
// For the latter, we have to check:
//   a) left sibling has node to spare
//   b) right sibling has node to spare
//   c) we must merge
// To simplify our code here, we handle cases #1 and #2 the same:
// If a node doesn't have enough items, we make sure it does (using a,b,c).
// We then simply redo our remove call, and the second time (regardless of
// whether we're in case 1 or 2), we'll have enough items and can guarantee
// that we hit case A.
func (n *node) growChildAndRemove(i int, key Key, minItems int, typ toRemove, less lessFunc) (item, bool) {
	if i > 0 && len(n.children[i-1].items) > minItems {
		// Steal from left child
		child := n.mutableChild(i)
		stealFrom := n.mutableChild(i - 1)
		stolenItem := stealFrom.items.pop()
		stealFrom.size--
		child.items.insertAt(0, n.items[i-1])
		child.size++
		n.items[i-1] = stolenItem
		if len(stealFrom.children) > 0 {
			c := stealFrom.children.pop()
			stealFrom.size -= c.size
			child.children.insertAt(0, c)
			child.size += c.size
		}
	} else if i < len(n.items) && len(n.children[i+1].items) > minItems {
		// steal from right child
		child := n.mutableChild(i)
		stealFrom := n.mutableChild(i + 1)
		stolenItem := stealFrom.items.removeAt(0)
		stealFrom.size--
		child.items = append(child.items, n.items[i])
		child.size++
		n.items[i] = stolenItem
		if len(stealFrom.children) > 0 {
			c := stealFrom.children.removeAt(0)
			stealFrom.size -= c.size
			child.children = append(child.children, c)
			child.size += c.size
		}
	} else {
		if i >= len(n.items) {
			i--
		}
		child := n.mutableChild(i)
		// merge with right child
		mergeItem := n.items.removeAt(i)
		mergeChild := n.children.removeAt(i + 1)
		child.items = append(child.items, mergeItem)
		child.items = append(child.items, mergeChild.items...)
		child.children = append(child.children, mergeChild.children...)
		child.size = child.computeSize()
		n.cow.freeNode(mergeChild)
	}
	return n.remove(key, minItems, typ, less)
}

// BTree is an implementation of a B-Tree.
//
// BTree stores item instances in an ordered structure, allowing easy insertion,
// removal, and iteration.
//
// Write operations are not safe for concurrent mutation by multiple
// goroutines, but Read operations are.
type BTree struct {
	degree int
	less   lessFunc
	root   *node
	cow    *copyOnWriteContext
}

// copyOnWriteContext pointers determine node ownership. A tree with a cow
// context equivalent to a node's cow context is allowed to modify that node.
// A tree whose write context does not match a node's is not allowed to modify
// it, and must create a new, writable copy (IE: it's a Clone).
//
// When doing any write operation, we maintain the invariant that the current
// node's context is equal to the context of the tree that requested the write.
// We do this by, before we descend into any node, creating a copy with the
// correct context if the contexts don't match.
//
// Since the node we're currently visiting on any write has the requesting
// tree's context, that node is modifiable in place.  Children of that node may
// not share context, but before we descend into them, we'll make a mutable
// copy.
type copyOnWriteContext struct{ byte } // non-empty, because empty structs may have same addr

// Clone clones the btree, lazily.  Clone should not be called concurrently,
// but the original tree (t) and the new tree (t2) can be used concurrently
// once the Clone call completes.
//
// The internal tree structure of b is marked read-only and shared between t and
// t2.  Writes to both t and t2 use copy-on-write logic, creating new nodes
// whenever one of b's original nodes would have been modified.  Read operations
// should have no performance degredation.  Write operations for both t and t2
// will initially experience minor slow-downs caused by additional allocs and
// copies due to the aforementioned copy-on-write logic, but should converge to
// the original performance characteristics of the original tree.
func (t *BTree) Clone() *BTree {
	// Create two entirely new copy-on-write contexts.
	// This operation effectively creates three trees:
	//   the original, shared nodes (old b.cow)
	//   the new b.cow nodes
	//   the new out.cow nodes
	cow1, cow2 := *t.cow, *t.cow
	out := *t
	t.cow = &cow1
	out.cow = &cow2
	return &out
}

// maxItems returns the max number of items to allow per node.
func (t *BTree) maxItems() int {
	return t.degree*2 - 1
}

// minItems returns the min number of items to allow per node (ignored for the
// root node).
func (t *BTree) minItems() int {
	return t.degree - 1
}

var nodePool = sync.Pool{New: func() interface{} { return new(node) }}

func (c *copyOnWriteContext) newNode() *node {
	n := nodePool.Get().(*node)
	n.cow = c
	return n
}

func (c *copyOnWriteContext) freeNode(n *node) {
	if n.cow == c {
		// clear to allow GC
		n.items.truncate(0)
		n.children.truncate(0)
		n.cow = nil
		nodePool.Put(n)
	}
}

// Set sets the given key to the given value in the tree. If the key is present in
// the tree, its value is changed and the old value is returned along with a second
// return value of true. If the key is not in the tree, it is added, and the second
// return value is false.
func (t *BTree) Set(k Key, v Value) (old Value, present bool) {
	old, present, _ = t.set(k, v, false)
	return old, present
}

// SetWithIndex sets the given key to the given value in the tree, and returns the
// index at which it was inserted.
func (t *BTree) SetWithIndex(k Key, v Value) (old Value, present bool, index int) {
	return t.set(k, v, true)
}

func (t *BTree) set(k Key, v Value, computeIndex bool) (old Value, present bool, idx int) {
	if t.root == nil {
		t.root = t.cow.newNode()
		t.root.items = append(t.root.items, item{k, v})
		t.root.size = 1
		return old, false, 0
	}
	t.root = t.root.mutableFor(t.cow)
	if len(t.root.items) >= t.maxItems() {
		sz := t.root.size
		item2, second := t.root.split(t.maxItems() / 2)
		oldroot := t.root
		t.root = t.cow.newNode()
		t.root.items = append(t.root.items, item2)
		t.root.children = append(t.root.children, oldroot, second)
		t.root.size = sz
	}

	return t.root.insert(item{k, v}, t.maxItems(), t.less, computeIndex)
}

// Delete removes the item with the given key, returning its value. The second return value
// reports whether the key was found.
func (t *BTree) Delete(k Key) (Value, bool) {
	m, removed := t.deleteItem(k, removeItem)
	return m.value, removed
}

// DeleteMin removes the smallest item in the tree and returns its key and value.
// If the tree is empty, it returns zero values.
func (t *BTree) DeleteMin() (Key, Value) {
	item, _ := t.deleteItem(nil, removeMin)
	return item.key, item.value
}

// DeleteMax removes the largest item in the tree and returns its key and value.
// If the tree is empty, it returns zero values.
func (t *BTree) DeleteMax() (Key, Value) {
	item, _ := t.deleteItem(nil, removeMax)
	return item.key, item.value
}

func (t *BTree) deleteItem(key Key, typ toRemove) (item, bool) {
	if t.root == nil || len(t.root.items) == 0 {
		return item{}, false
	}
	t.root = t.root.mutableFor(t.cow)
	out, removed := t.root.remove(key, t.minItems(), typ, t.less)
	if len(t.root.items) == 0 && len(t.root.children) > 0 {
		oldroot := t.root
		t.root = t.root.children[0]
		t.cow.freeNode(oldroot)
	}
	return out, removed
}

// Get returns the value for the given key in the tree, or the zero value if the
// key is not in the tree.
//
// To distinguish a zero value from a key that is not present, use GetWithIndex.
func (t *BTree) Get(k Key) Value {
	var z Value
	if t.root == nil {
		return z
	}
	item, ok, _ := t.root.get(k, false, t.less)
	if !ok {
		return z
	}
	return item.value
}

// GetWithIndex returns the value and index for the given key in the tree, or the
// zero value and -1 if the key is not in the tree.
func (t *BTree) GetWithIndex(k Key) (Value, int) {
	var z Value
	if t.root == nil {
		return z, -1
	}
	item, _, index := t.root.get(k, true, t.less)
	return item.value, index
}

// At returns the key and value at index i. The minimum item has index 0.
// If i is outside the range [0, t.Len()), At panics.
func (t *BTree) At(i int) (Key, Value) {
	if i < 0 || i >= t.Len() {
		panic("btree: index out of range")
	}
	item := t.root.at(i)
	return item.key, item.value
}

// Has reports whether the given key is in the tree.
func (t *BTree) Has(k Key) bool {
	if t.root == nil {
		return false
	}
	_, ok, _ := t.root.get(k, false, t.less)
	return ok
}

// Min returns the smallest key in the tree and its value. If the tree is empty, it
// returns zero values.
func (t *BTree) Min() (Key, Value) {
	var k Key
	var v Value
	if t.root == nil {
		return k, v
	}
	n := t.root
	for len(n.children) > 0 {
		n = n.children[0]
	}
	if len(n.items) == 0 {
		return k, v
	}
	return n.items[0].key, n.items[0].value
}

// Max returns the largest key in the tree and its value. If the tree is empty, both
// return values are zero values.
func (t *BTree) Max() (Key, Value) {
	var k Key
	var v Value
	if t.root == nil {
		return k, v
	}
	n := t.root
	for len(n.children) > 0 {
		n = n.children[len(n.children)-1]
	}
	if len(n.items) == 0 {
		return k, v
	}
	m := n.items[len(n.items)-1]
	return m.key, m.value
}

// Len returns the number of items currently in the tree.
func (t *BTree) Len() int {
	if t.root == nil {
		return 0
	}
	return t.root.size
}

// Before returns an iterator positioned just before k. After the first call to Next,
// the Iterator will be at k, or at the key just greater than k if k is not in the tree.
// Subsequent calls to Next will traverse the tree's items in ascending order.
func (t *BTree) Before(k Key) *Iterator {
	if t.root == nil {
		return &Iterator{}
	}
	var cs cursorStack
	cs, found, idx := t.root.cursorStackForKey(k, cs, t.less)
	// If we found the key, the cursor stack is pointing to it. Since that is
	// the first element we want, don't advance the iterator on the initial call to Next.
	// If we haven't found the key, then the top of the cursor stack is either pointing at the
	// item just after k, in which case we do not want to move the iterator; or the index
	// is past the end of the items slice, in which case we do.
	var stay bool
	top := cs[len(cs)-1]
	if found {
		stay = true
	} else if top.index < len(top.node.items) {
		stay = true
	} else {
		idx--
	}
	return &Iterator{
		cursors:    cs,
		stay:       stay,
		descending: false,
		Index:      idx,
	}
}

// After returns an iterator positioned just after k. After the first call to Next,
// the Iterator will be at k, or at the key just less than k if k is not in the tree.
// Subsequent calls to Next will traverse the tree's items in descending order.
func (t *BTree) After(k Key) *Iterator {
	if t.root == nil {
		return &Iterator{}
	}
	var cs cursorStack
	cs, found, idx := t.root.cursorStackForKey(k, cs, t.less)
	// If we found the key, the cursor stack is pointing to it. Since that is
	// the first element we want, don't advance the iterator on the initial call to Next.
	// If we haven't found the key, the cursor stack is pointing just after the first item,
	// so we do want to advance.
	return &Iterator{
		cursors:    cs,
		stay:       found,
		descending: true,
		Index:      idx,
	}
}

// BeforeIndex returns an iterator positioned just before the item with the given index.
// The iterator will traverse the tree's items in ascending order.
// If i is not in the range [0, tr.Len()], BeforeIndex panics.
// Note that it is not an error to provide an index of tr.Len().
func (t *BTree) BeforeIndex(i int) *Iterator {
	return t.indexIterator(i, false)
}

// AfterIndex returns an iterator positioned just after the item with the given index.
// The iterator will traverse the tree's items in descending order.
// If i is not in the range [0, tr.Len()], AfterIndex panics.
// Note that it is not an error to provide an index of tr.Len().
func (t *BTree) AfterIndex(i int) *Iterator {
	return t.indexIterator(i, true)
}

func (t *BTree) indexIterator(i int, descending bool) *Iterator {
	if i < 0 || i > t.Len() {
		panic("btree: index out of range")
	}
	if i == t.Len() {
		return &Iterator{}
	}
	var cs cursorStack
	return &Iterator{
		cursors:    t.root.cursorStackForIndex(i, cs),
		stay:       true,
		descending: descending,
		Index:      i,
	}
}

// An Iterator supports traversing the items in the tree.
type Iterator struct {
	Key   Key
	Value Value
	// Index is the position of the item in the tree viewed as a sequence.
	// The minimum item has index zero.
	Index int

	cursors    cursorStack // stack of nodes with indices; last element is the top
	stay       bool        // don't do anything on the first call to Next.
	descending bool        // traverse the items in descending order
}

// Next advances the Iterator to the next item in the tree. If Next returns true,
// the Iterator's Key, Value and Index fields refer to the next item. If Next returns
// false, there are no more items and the values of Key, Value and Index are undefined.
//
// If the tree is modified during iteration, the behavior is undefined.
func (it *Iterator) Next() bool {
	var more bool
	switch {
	case len(it.cursors) == 0:
		more = false
	case it.stay:
		it.stay = false
		more = true
	case it.descending:
		more = it.dec()
	default:
		more = it.inc()
	}
	if !more {
		return false
	}
	top := it.cursors[len(it.cursors)-1]
	item := top.node.items[top.index]
	it.Key = item.key
	it.Value = item.value
	return true
}

// When inc returns true, the top cursor on the stack refers to the new current item.
func (it *Iterator) inc() bool {
	// Useful invariants for understanding this function:
	// - Leaf nodes have zero children, and zero or more items.
	// - Nonleaf nodes have one more child than item, and children[i] < items[i] < children[i+1].
	// - The current item in the iterator is top.node.items[top.index].

	it.Index++
	// If we are at a non-leaf node, the current item is items[i], so
	// now we want to continue with children[i+1], which must exist
	// by the node invariant. We want the minimum item in that child's subtree.
	top := it.cursors.incTop(1)
	for len(top.node.children) > 0 {
		top = cursor{top.node.children[top.index], 0}
		it.cursors.push(top)
	}
	// Here, we are at a leaf node. top.index points to
	// the new current item, if it's within the items slice.
	for top.index >= len(top.node.items) {
		// We've gone through everything in this node. Pop it off the stack.
		it.cursors.pop()
		// If the stack is now empty,we're past the last item in the tree.
		if it.cursors.empty() {
			return false
		}
		top = it.cursors.top()
		// The new top's index points to a child, which we've just finished
		// exploring. The next item is the one at the same index in the items slice.
	}
	// Here, the top cursor on the stack points to the new current item.
	return true
}

func (it *Iterator) dec() bool {
	// See the invariants for inc, above.
	it.Index--
	top := it.cursors.top()
	// If we are at a non-leaf node, the current item is items[i], so
	// now we want to continue with children[i]. We want the maximum item in that child's subtree.
	for len(top.node.children) > 0 {
		c := top.node.children[top.index]
		top = cursor{c, len(c.items)}
		it.cursors.push(top)
	}
	top = it.cursors.incTop(-1)
	// Here, we are at a leaf node. top.index points to
	// the new current item, if it's within the items slice.
	for top.index < 0 {
		// We've gone through everything in this node. Pop it off the stack.
		it.cursors.pop()
		// If the stack is now empty,we're past the last item in the tree.
		if it.cursors.empty() {
			return false
		}
		// The new top's index points to a child, which we've just finished
		// exploring. That child is to the right of the item we want to advance to,
		// so decrement the index.
		top = it.cursors.incTop(-1)
	}
	return true
}

// A cursor is effectively a pointer into a node. A stack of cursors identifies an item in the tree,
// and makes it possible to move to the next or previous item efficiently.
//
// If the cursor is on the top of the stack, its index points into the node's items slice, selecting
// the current item. Otherwise, the index points into the children slice and identifies the child
// that is next in the stack.
type cursor struct {
	node  *node
	index int
}

// A cursorStack is a stack of cursors, representing a path of nodes from the root of the tree.
type cursorStack []cursor

func (s *cursorStack) push(c cursor) cursorStack {
	*s = append(*s, c)
	return *s
}

func (s *cursorStack) pop() cursor {
	last := len(*s) - 1
	t := (*s)[last]
	*s = (*s)[:last]
	return t
}

func (s *cursorStack) top() cursor {
	return (*s)[len(*s)-1]
}

func (s *cursorStack) empty() bool {
	return len(*s) == 0
}

// incTop increments top's index by n and returns it.
func (s *cursorStack) incTop(n int) cursor {
	(*s)[len(*s)-1].index += n // Don't call top: modify the original, not a copy.
	return s.top()
}
