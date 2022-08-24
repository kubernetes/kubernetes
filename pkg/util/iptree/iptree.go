/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package iptree

import (
	"fmt"
	"math/bits"
	"net/netip"
	"reflect"
)

// based on https://github.com/armon/go-radix but
// the tree is binary to simplify implementation.

// node is an element of radix tree with a netip.Prefix optimized to store IP Addresses.
type node struct {
	// prefix network CIDR
	prefix netip.Prefix

	// leaf is used to store value
	leaf bool
	val  interface{}

	child [2]*node
}

func (n *node) isLeaf() bool {
	return n.leaf
}

func (n *node) mergeChild() {
	// can not merge if there are two or no childs
	if n.child[0] != nil &&
		n.child[1] != nil {
		return
	}
	if n.child[0] == nil &&
		n.child[1] == nil {
		return
	}
	// find the child and merge it
	var idx int
	if n.child[0] != nil {
		idx = 0
	} else if n.child[1] != nil {
		idx = 1
	}
	child := n.child[idx]
	n.prefix = child.prefix
	n.leaf = child.leaf
	n.val = child.val
	n.child = child.child
}

// Tree is a radix tree for IPv4 and IPv6 networks.
type Tree struct {
	root *node
	is6  bool
	size int
}

// New creates a new Radix Tree for IP addresses
func New(is6 bool) *Tree {
	zero := netip.IPv4Unspecified()
	if is6 {
		zero = netip.IPv6Unspecified()
	}
	return &Tree{
		root: &node{
			prefix: netip.PrefixFrom(zero, 0),
		},
		is6: is6,
	}
}

func (t *Tree) Len() int {
	return t.size
}

func (t *Tree) Get(subnet string) (interface{}, bool) {
	prefix, err := netip.ParsePrefix(subnet)
	if err != nil {
		return nil, false
	}
	return t.getPrefix(prefix)
}

// Get returns true if the prefix exists in the tree
func (t *Tree) getPrefix(prefix netip.Prefix) (interface{}, bool) {
	if prefix.Addr().Is6() != t.is6 {
		return nil, false
	}
	n := t.root
	bitPosition := 0
	// mask the address for sanity
	address := prefix.Masked().Addr()
	// we can't check longer than the request mask
	mask := prefix.Bits()
	for bitPosition <= mask {
		// same mask, return node if it contains a value
		// we already validated the suffixes match
		if bitPosition == mask {
			if n.isLeaf() {
				return n.val, true
			}
			break
		}

		// Look for a child checking the bit position after the mask
		n = n.child[getBitFromAddr(address, bitPosition+1)]
		if n == nil {
			break
		}

		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()

		// check we are in the right branch comparing the suffixes
		if !n.prefix.Contains(address) {
			break
		}

	}
	return nil, false

}

// LongestPrefixMatch returns the longest prefix match, the stored value and true if exist.
func (t *Tree) LongestPrefixMatch(subnet string) (string, interface{}, bool) {
	prefix, err := netip.ParsePrefix(subnet)
	if err != nil {
		return "", nil, false
	}
	p, v, ok := t.longestPrefix(prefix)
	if !ok {
		return "", nil, false
	}
	return p.String(), v, ok
}

func (t *Tree) longestPrefix(prefix netip.Prefix) (netip.Prefix, interface{}, bool) {
	if prefix.Addr().Is6() != t.is6 {
		return netip.Prefix{}, nil, false
	}

	var last *node
	n := t.root
	// bit position is given by the mask bits
	bitPosition := 0
	// mask the address
	address := prefix.Masked().Addr()
	mask := prefix.Bits()
	for bitPosition <= mask {
		if n.isLeaf() {
			last = n
		}
		// we branch at the mask bit
		if bitPosition == mask {
			break
		}

		// Look for a child checking the bit position after the mask
		n = n.child[getBitFromAddr(address, bitPosition+1)]
		if n == nil {
			break
		}

		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()

		// check we are in the right branch
		if !n.prefix.Contains(address) {
			break
		}

	}
	if last != nil {
		return last.prefix, last.val, true
	}
	return netip.Prefix{}, nil, false

}

// ShortestPrefixMatch returns the shortest prefix match, the stored value and true if exist.
func (t *Tree) ShortestPrefixMatch(subnet string) (string, interface{}, bool) {
	prefix, err := netip.ParsePrefix(subnet)
	if err != nil {
		return "", nil, false
	}
	p, v, ok := t.shortestPrefix(prefix)
	if !ok {
		return "", nil, false
	}
	return p.String(), v, ok
}

func (t *Tree) shortestPrefix(prefix netip.Prefix) (netip.Prefix, interface{}, bool) {
	if prefix.Addr().Is6() != t.is6 {
		return netip.Prefix{}, nil, false
	}

	n := t.root
	// bit position is given by the mask bits
	bitPosition := 0
	// mask the address
	address := prefix.Masked().Addr()
	mask := prefix.Bits()
	for bitPosition <= mask {
		if n.isLeaf() {
			return n.prefix, n.val, true
		}
		// we branch at the mask bit
		if bitPosition == mask {
			break
		}

		// Look for a child checking the bit position after the mask
		n = n.child[getBitFromAddr(address, bitPosition+1)]
		if n == nil {
			break
		}

		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()

		// check we are in the right branch
		if !n.prefix.Contains(address) {
			break
		}

	}

	return netip.Prefix{}, nil, false

}

func (t *Tree) Insert(subnet string, v interface{}) (interface{}, bool) {
	prefix, err := netip.ParsePrefix(subnet)
	if err != nil {
		return nil, false
	}
	return t.insertPrefix(prefix, v)
}

// Insert is used to add a newentry or update
// an existing entry. Returns true if updated.
func (t *Tree) insertPrefix(prefix netip.Prefix, v interface{}) (interface{}, bool) {
	if prefix.Addr().Is6() != t.is6 {
		return nil, false
	}
	var parent *node
	n := t.root
	// bit position is given by the mask bits
	bitPosition := 0
	// mask the address
	address := prefix.Masked().Addr()
	mask := prefix.Bits()
	for bitPosition <= mask {
		// found it
		if bitPosition == mask {
			// replace and return the old value
			if n.isLeaf() {
				old := n.val
				n.val = v
				n.leaf = true
				return old, true
			}

			n.leaf = true
			n.val = v
			t.size++
			return nil, false
		}

		// Look for a child checking the bit position after the mask
		childIndex := getBitFromAddr(address, bitPosition+1)
		parent = n
		n = n.child[childIndex]

		// No child, create one
		if n == nil {
			parent.child[childIndex] = &node{
				leaf:   true,
				val:    v,
				prefix: prefix,
			}

			t.size++
			return nil, false
		}

		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()

		// continue if we are in the right branch and current
		// node is our parent
		if n.prefix.Contains(address) && bitPosition <= mask {
			continue
		}

		// Split the node and add a new child:
		// - Case 1: parent -> child -> n
		// - Case 2: parent -> newnode |--> child
		//                             |--> n

		t.size++
		child := &node{
			prefix: prefix,
			leaf:   true,
			val:    v,
		}
		// Case 1: existing node is a sibling
		if prefix.Contains(n.prefix.Addr()) && bitPosition > mask {
			// parent to child
			parent.child[childIndex] = child
			pos := prefix.Bits() + 1
			// calculate if the sibling is at the left or right
			child.child[getBitFromAddr(n.prefix.Addr(), pos)] = n
			return nil, false
		}
		// Case 2: existing node has the same mask but different based address
		// add common ancestor and branch on it
		ancestor := findAncestor(prefix, n.prefix)
		link := &node{
			prefix: ancestor,
		}
		pos := parent.prefix.Bits() + 1
		parent.child[getBitFromAddr(ancestor.Addr(), pos)] = link
		// ancestor -> childs
		pos = ancestor.Bits() + 1
		idxChild := getBitFromAddr(prefix.Addr(), pos)
		idxN := getBitFromAddr(n.prefix.Addr(), pos)
		if idxChild == idxN {
			panic(fmt.Sprintf("wrong ancestor %s: child %s N %s", ancestor.String(), prefix.String(), n.prefix.String()))
		}
		link.child[idxChild] = child
		link.child[idxN] = n
		return nil, false
	}

	return nil, false
}

// Delete subnet and return the stored value and true if the subnet exist
func (t *Tree) Delete(subnet string) (interface{}, bool) {
	prefix, err := netip.ParsePrefix(subnet)
	if err != nil {
		return nil, false
	}
	return t.deletePrefix(prefix)
}

func (t *Tree) deletePrefix(prefix netip.Prefix) (interface{}, bool) {
	if prefix.Addr().Is6() != t.is6 {
		return nil, false
	}
	var parent *node
	n := t.root
	// bit position is given by the mask bits
	bitPosition := 0
	// mask the address
	address := prefix.Masked().Addr()
	mask := prefix.Bits()
	for bitPosition <= mask {
		// found it
		if bitPosition == mask {
			if !n.isLeaf() {
				return nil, false
			}
			goto DELETE
		}

		// Look for a child checking the bit position after the mask
		childIndex := getBitFromAddr(address, bitPosition+1)
		parent = n
		n = n.child[childIndex]
		if n == nil {
			return nil, false
		}

		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()

		// check we are in the right branch
		if !n.prefix.Contains(address) {
			return nil, false
		}
	}
	return nil, false

DELETE:
	// Delete the value
	n.leaf = false
	val := n.val
	t.size--

	nodeChildren := 0
	if n.child[0] != nil {
		nodeChildren++
	}
	if n.child[1] != nil {
		nodeChildren++
	}

	// Check if we should delete this node from the parent
	if parent != nil && nodeChildren == 0 {
		if parent.child[0] != nil && parent.child[0] == n {
			parent.child[0] = nil
		} else if parent.child[1] != nil && parent.child[1] == n {
			parent.child[1] = nil
		} else {
			panic("wrong parent")
		}
		n = nil
	}

	// Check if we should merge this node
	if n != t.root && nodeChildren == 1 {
		n.mergeChild()
	}

	// Check if we should merge the parent's other child
	parentChildren := 0
	if parent != nil {
		if parent.child[0] != nil {
			parentChildren++
		}
		if parent.child[1] != nil {
			parentChildren++
		}
		if parent != t.root && parentChildren == 1 && !parent.isLeaf() {
			parent.mergeChild()
		}
	}

	return val, true
}

// WalkFn is used when walking the tree. Takes a
// key and value, returning if iteration should
// be terminated.
type WalkFn func(s netip.Prefix, v interface{}) bool

// Walk is used to walk the tree
func (t *Tree) Walk(fn WalkFn) {
	recursiveWalk(t.root, fn)
}

// recursiveWalk is used to do a pre-order walk of a node
// recursively. Returns true if the walk should be aborted
func recursiveWalk(n *node, fn WalkFn) bool {
	// Visit the leaf values if any
	if n.leaf && fn(n.prefix, n.val) {
		return true
	}

	// Recurse on the children
	if n.child[0] != nil {
		if recursiveWalk(n.child[0], fn) {
			return true
		}
	}
	if n.child[1] != nil {
		if recursiveWalk(n.child[1], fn) {
			return true
		}
	}
	return false
}

// ToMap is used to walk the tree and convert it into a map
func (t *Tree) ToMap() map[string]interface{} {
	out := make(map[string]interface{}, t.size)
	t.Walk(func(k netip.Prefix, v interface{}) bool {
		out[k.String()] = v
		return false
	})
	return out
}

// Equal returns true if both trees conain the same keys and values
func (t *Tree) Equal(tree *Tree) bool {
	if t.Len() != tree.Len() {
		return false
	}

	m1 := t.ToMap()
	m2 := tree.ToMap()
	return reflect.DeepEqual(m1, m2)
}

// Parents return the Parents nodes in a tree
func (t *Tree) Parents() map[string]interface{} {
	result := map[string]interface{}{}
	queue := []*node{t.root}

	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		// store and continue, only interested on the parents
		if n.isLeaf() {
			result[n.prefix.String()] = n.val
			continue
		}
		if n.child[0] != nil {
			queue = append(queue, n.child[0])
		}
		if n.child[1] != nil {
			queue = append(queue, n.child[1])
		}
	}
	return result
}

// assume starts at 0 from the MSB: 0.1.2......31
// return 0 or 1
func getBitFromAddr(ip netip.Addr, pos int) int {
	bytes := ip.AsSlice()
	// get the byte in the slice
	index := (pos - 1) / 8
	if index >= len(bytes) {
		panic(fmt.Sprintf("ip %s pos %d index %d bytes %v", ip, pos, index, bytes))
	}
	// get the offset inside the byte
	offset := (pos - 1) % 8
	// check if the bit is set
	if bytes[index]&(uint8(0x80)>>offset) > 0 {
		return 1
	} else {
		return 0
	}

}

// find the common subnet, aka the one with the common prefix
func findAncestor(a, b netip.Prefix) netip.Prefix {
	bytesA := a.Addr().AsSlice()
	bytesB := b.Addr().AsSlice()
	bytes := make([]byte, len(bytesA))

	max := a.Bits()
	if l := b.Bits(); l < max {
		max = l
	}

	mask := 0
	for i := range bytesA {
		xor := bytesA[i] ^ bytesB[i]
		if xor == 0 {
			bytes[i] = bytesA[i]
			mask += 8

		} else {
			pos := bits.LeadingZeros8(xor)
			mask += pos
			// mask off the non leading zeros
			bytes[i] = bytesA[i] & (^uint8(0) << (8 - pos))
			break
		}
	}
	if mask > max {
		mask = max
	}

	addr, ok := netip.AddrFromSlice(bytes)
	if !ok {
		panic(bytes)
	}
	ancestor := netip.PrefixFrom(addr, mask)
	return ancestor.Masked()

}
