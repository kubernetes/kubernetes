/*
Copyright 2023 The Kubernetes Authors.

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
)

// iptree implement a radix tree that uses IP prefixes as nodes and allows to store values in each node.
// Example:
//
// 	r := New[int]()
//
//	prefixes := []string{
//		"0.0.0.0/0",
//		"10.0.0.0/8",
//		"10.0.0.0/16",
//		"10.1.0.0/16",
//		"10.1.1.0/24",
//		"10.1.244.0/24",
//		"10.0.0.0/24",
//		"10.0.0.3/32",
//		"192.168.0.0/24",
//		"192.168.0.0/28",
//		"192.168.129.0/28",
//	}
//	for _, k := range prefixes {
//		r.InsertPrefix(netip.MustParsePrefix(k), 0)
//	}
//
// (*) means the node is not public, is not storing any value
//
// 0.0.0.0/0 --- 10.0.0.0/8 --- *10.0.0.0/15 --- 10.0.0.0/16 --- 10.0.0.0/24 --- 10.0.0.3/32
//  |                                 |
//  |                                 \ -------- 10.1.0.0/16 --- 10.1.1.0/24
//  |                                                 |
//  |	                                               \ ------- 10.1.244.0/24
//  |
//  \------ *192.168.0.0/16 --- 192.168.0.0/24 --- 192.168.0.0/28
//                   |
//                    \ -------- 192.168.129.0/28

// node is an element of radix tree with a netip.Prefix optimized to store IP prefixes.
type node[T any] struct {
	// prefix network CIDR
	prefix netip.Prefix
	// public nodes are used to store values
	public bool
	val    T

	child [2]*node[T] // binary tree
}

// mergeChild allow to compress the tree
// when n has exactly one child and no value
// p -> n -> b -> c ==> p -> b -> c
func (n *node[T]) mergeChild() {
	// public nodes can not be merged
	if n.public {
		return
	}
	// can not merge if there are two children
	if n.child[0] != nil &&
		n.child[1] != nil {
		return
	}
	// can not merge if there are no children
	if n.child[0] == nil &&
		n.child[1] == nil {
		return
	}
	// find the child and merge it
	var child *node[T]
	if n.child[0] != nil {
		child = n.child[0]
	} else if n.child[1] != nil {
		child = n.child[1]
	}
	n.prefix = child.prefix
	n.public = child.public
	n.val = child.val
	n.child = child.child
	// remove any references from the deleted node
	// to avoid memory leak
	child.child[0] = nil
	child.child[1] = nil
}

// Tree is a radix tree for IPv4 and IPv6 networks.
type Tree[T any] struct {
	rootV4 *node[T]
	rootV6 *node[T]
}

// New creates a new Radix Tree for IP addresses.
func New[T any]() *Tree[T] {
	return &Tree[T]{
		rootV4: &node[T]{
			prefix: netip.PrefixFrom(netip.IPv4Unspecified(), 0),
		},
		rootV6: &node[T]{
			prefix: netip.PrefixFrom(netip.IPv6Unspecified(), 0),
		},
	}
}

// GetPrefix returns the stored value and true if the exact prefix exists in the tree.
func (t *Tree[T]) GetPrefix(prefix netip.Prefix) (T, bool) {
	var zeroT T

	n := t.rootV4
	if prefix.Addr().Is6() {
		n = t.rootV6
	}
	bitPosition := 0
	// mask the address for sanity
	address := prefix.Masked().Addr()
	// we can't check longer than the request mask
	mask := prefix.Bits()
	// walk the network bits of the prefix
	for bitPosition < mask {
		// Look for a child checking the bit position after the mask
		n = n.child[getBitFromAddr(address, bitPosition+1)]
		if n == nil {
			return zeroT, false
		}
		// check we are in the right branch comparing the suffixes
		if !n.prefix.Contains(address) {
			return zeroT, false
		}
		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()
	}
	// check if this node is a public node and contains a prefix
	if n != nil && n.public && n.prefix == prefix {
		return n.val, true
	}

	return zeroT, false
}

// LongestPrefixMatch returns the longest prefix match, the stored value and true if exist.
// For example, considering the following prefixes 192.168.20.16/28 and 192.168.0.0/16,
// when the address 192.168.20.19/32 is looked up it will return 192.168.20.16/28.
func (t *Tree[T]) LongestPrefixMatch(prefix netip.Prefix) (netip.Prefix, T, bool) {
	n := t.rootV4
	if prefix.Addr().Is6() {
		n = t.rootV6
	}

	var last *node[T]
	// bit position is given by the mask bits
	bitPosition := 0
	// mask the address
	address := prefix.Masked().Addr()
	mask := prefix.Bits()
	// walk the network bits of the prefix
	for bitPosition < mask {
		if n.public {
			last = n
		}
		// Look for a child checking the bit position after the mask
		n = n.child[getBitFromAddr(address, bitPosition+1)]
		if n == nil {
			break
		}
		// check we are in the right branch comparing the suffixes
		if !n.prefix.Contains(address) {
			break
		}
		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()
	}

	if n != nil && n.public && n.prefix == prefix {
		last = n
	}

	if last != nil {
		return last.prefix, last.val, true
	}
	var zeroT T
	return netip.Prefix{}, zeroT, false
}

// ShortestPrefixMatch returns the shortest prefix match, the stored value and true if exist.
// For example, considering the following prefixes 192.168.20.16/28 and 192.168.0.0/16,
// when the address 192.168.20.19/32 is looked up it will return 192.168.0.0/16.
func (t *Tree[T]) ShortestPrefixMatch(prefix netip.Prefix) (netip.Prefix, T, bool) {
	var zeroT T

	n := t.rootV4
	if prefix.Addr().Is6() {
		n = t.rootV6
	}
	// bit position is given by the mask bits
	bitPosition := 0
	// mask the address
	address := prefix.Masked().Addr()
	mask := prefix.Bits()
	for bitPosition < mask {
		if n.public {
			return n.prefix, n.val, true
		}
		// Look for a child checking the bit position after the mask
		n = n.child[getBitFromAddr(address, bitPosition+1)]
		if n == nil {
			return netip.Prefix{}, zeroT, false
		}
		// check we are in the right branch comparing the suffixes
		if !n.prefix.Contains(address) {
			return netip.Prefix{}, zeroT, false
		}
		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()
	}

	if n != nil && n.public && n.prefix == prefix {
		return n.prefix, n.val, true
	}
	return netip.Prefix{}, zeroT, false
}

// InsertPrefix is used to add a new entry or update
// an existing entry. Returns true if updated.
func (t *Tree[T]) InsertPrefix(prefix netip.Prefix, v T) bool {
	n := t.rootV4
	if prefix.Addr().Is6() {
		n = t.rootV6
	}
	var parent *node[T]
	// bit position is given by the mask bits
	bitPosition := 0
	// mask the address
	address := prefix.Masked().Addr()
	mask := prefix.Bits()
	for bitPosition < mask {
		// Look for a child checking the bit position after the mask
		childIndex := getBitFromAddr(address, bitPosition+1)
		parent = n
		n = n.child[childIndex]
		// if no child create a new one with
		if n == nil {
			parent.child[childIndex] = &node[T]{
				public: true,
				val:    v,
				prefix: prefix,
			}
			return false
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
		child := &node[T]{
			prefix: prefix,
			public: true,
			val:    v,
		}
		// Case 1: existing node is a sibling
		if prefix.Contains(n.prefix.Addr()) && bitPosition > mask {
			// parent to child
			parent.child[childIndex] = child
			pos := prefix.Bits() + 1
			// calculate if the sibling is at the left or right
			child.child[getBitFromAddr(n.prefix.Addr(), pos)] = n
			return false
		}

		// Case 2: existing node has the same mask but different base address
		// add common ancestor and branch on it
		ancestor := findAncestor(prefix, n.prefix)
		link := &node[T]{
			prefix: ancestor,
		}
		pos := parent.prefix.Bits() + 1
		parent.child[getBitFromAddr(ancestor.Addr(), pos)] = link
		// ancestor -> children
		pos = ancestor.Bits() + 1
		idxChild := getBitFromAddr(prefix.Addr(), pos)
		idxN := getBitFromAddr(n.prefix.Addr(), pos)
		if idxChild == idxN {
			panic(fmt.Sprintf("wrong ancestor %s: child %s N %s", ancestor.String(), prefix.String(), n.prefix.String()))
		}
		link.child[idxChild] = child
		link.child[idxN] = n
		return false
	}

	// if already exist update it and make it public
	if n != nil && n.prefix == prefix {
		if n.public {
			n.val = v
			n.public = true
			return true
		}
		n.val = v
		n.public = true
		return false
	}

	return false
}

// DeletePrefix delete the exact prefix and return true if it existed.
func (t *Tree[T]) DeletePrefix(prefix netip.Prefix) bool {
	root := t.rootV4
	if prefix.Addr().Is6() {
		root = t.rootV6
	}
	var parent *node[T]
	n := root
	// bit position is given by the mask bits
	bitPosition := 0
	// mask the address
	address := prefix.Masked().Addr()
	mask := prefix.Bits()
	for bitPosition < mask {
		// Look for a child checking the bit position after the mask
		parent = n
		n = n.child[getBitFromAddr(address, bitPosition+1)]
		if n == nil {
			return false
		}
		// check we are in the right branch comparing the suffixes
		if !n.prefix.Contains(address) {
			return false
		}
		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()
	}
	// check if the node contains the prefix we want to delete
	if n.prefix != prefix {
		return false
	}
	// Delete the value
	n.public = false
	var zeroT T
	n.val = zeroT

	nodeChildren := 0
	if n.child[0] != nil {
		nodeChildren++
	}
	if n.child[1] != nil {
		nodeChildren++
	}
	// If there is a parent and this node does not have any children
	// this is a leaf so we can delete this node.
	// - parent -> child(to be deleted)
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
	// The root node can not be merged
	if n != root && nodeChildren == 1 {
		n.mergeChild()
	}
	// Check if we should merge the parent's other child
	// parent -> deletedNode
	//        |--> child
	parentChildren := 0
	if parent != nil {
		if parent.child[0] != nil {
			parentChildren++
		}
		if parent.child[1] != nil {
			parentChildren++
		}
		if parent != root && parentChildren == 1 && !parent.public {
			parent.mergeChild()
		}
	}
	return true
}

// for testing, returns the number of public nodes in the tree.
func (t *Tree[T]) Len(isV6 bool) int {
	count := 0
	t.DepthFirstWalk(isV6, func(k netip.Prefix, v T) bool {
		count++
		return false
	})
	return count
}

// WalkFn is used when walking the tree. Takes a
// key and value, returning if iteration should
// be terminated.
type WalkFn[T any] func(s netip.Prefix, v T) bool

// DepthFirstWalk is used to walk the tree of the corresponding IP family
func (t *Tree[T]) DepthFirstWalk(isIPv6 bool, fn WalkFn[T]) {
	if isIPv6 {
		recursiveWalk(t.rootV6, fn)
	}
	recursiveWalk(t.rootV4, fn)
}

// recursiveWalk is used to do a pre-order walk of a node
// recursively. Returns true if the walk should be aborted
func recursiveWalk[T any](n *node[T], fn WalkFn[T]) bool {
	if n == nil {
		return true
	}
	// Visit the public values if any
	if n.public && fn(n.prefix, n.val) {
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

// WalkPrefix is used to walk the tree under a prefix
func (t *Tree[T]) WalkPrefix(prefix netip.Prefix, fn WalkFn[T]) {
	n := t.rootV4
	if prefix.Addr().Is6() {
		n = t.rootV6
	}
	bitPosition := 0
	// mask the address for sanity
	address := prefix.Masked().Addr()
	// we can't check longer than the request mask
	mask := prefix.Bits()
	// walk the network bits of the prefix
	for bitPosition < mask {
		// Look for a child checking the bit position after the mask
		n = n.child[getBitFromAddr(address, bitPosition+1)]
		if n == nil {
			return
		}
		// check we are in the right branch comparing the suffixes
		if !n.prefix.Contains(address) {
			break
		}
		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()
	}
	recursiveWalk[T](n, fn)

}

// WalkPath is used to walk the tree, but only visiting nodes
// from the root down to a given IP prefix. Where WalkPrefix walks
// all the entries *under* the given prefix, this walks the
// entries *above* the given prefix.
func (t *Tree[T]) WalkPath(path netip.Prefix, fn WalkFn[T]) {
	n := t.rootV4
	if path.Addr().Is6() {
		n = t.rootV6
	}
	bitPosition := 0
	// mask the address for sanity
	address := path.Masked().Addr()
	// we can't check longer than the request mask
	mask := path.Bits()
	// walk the network bits of the prefix
	for bitPosition < mask {
		// Visit the public values if any
		if n.public && fn(n.prefix, n.val) {
			return
		}
		// Look for a child checking the bit position after the mask
		n = n.child[getBitFromAddr(address, bitPosition+1)]
		if n == nil {
			return
		}
		// check we are in the right branch comparing the suffixes
		if !n.prefix.Contains(address) {
			return
		}
		// update the new bit position with the new node mask
		bitPosition = n.prefix.Bits()
	}
	// check if this node is a public node and contains a prefix
	if n != nil && n.public && n.prefix == path {
		fn(n.prefix, n.val)
	}
}

// TopLevelPrefixes is used to return a map with all the Top Level prefixes
// from the corresponding IP family and its values.
// For example, if the tree contains entries for 10.0.0.0/8, 10.1.0.0/16, and 192.168.0.0/16,
// this will return 10.0.0.0/8 and 192.168.0.0/16.
func (t *Tree[T]) TopLevelPrefixes(isIPv6 bool) map[string]T {
	if isIPv6 {
		return t.topLevelPrefixes(t.rootV6)
	}
	return t.topLevelPrefixes(t.rootV4)
}

// topLevelPrefixes is used to return a map with all the Top Level prefixes and its values
func (t *Tree[T]) topLevelPrefixes(root *node[T]) map[string]T {
	result := map[string]T{}
	queue := []*node[T]{root}

	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		// store and continue, only interested on the top level prefixes
		if n.public {
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

// GetHostIPPrefixMatches returns the list of prefixes that contain the specified Host IP.
// An IP is considered a Host IP if is within the subnet range and is not the network address
// or, if IPv4, the broadcast address (RFC 1878).
func (t *Tree[T]) GetHostIPPrefixMatches(ip netip.Addr) map[netip.Prefix]T {
	// walk the tree to find all the prefixes containing this IP
	ipPrefix := netip.PrefixFrom(ip, ip.BitLen())
	prefixes := map[netip.Prefix]T{}
	t.WalkPath(ipPrefix, func(k netip.Prefix, v T) bool {
		if prefixContainIP(k, ipPrefix.Addr()) {
			prefixes[k] = v
		}
		return false
	})
	return prefixes
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
	}
	return 0
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

// prefixContainIP returns true if the given IP is contained with the prefix,
// is not the network address and also, if IPv4, is not the broadcast address.
// This is required because the Kubernetes allocators reserve these addresses
// so IPAddresses can not block deletion of this ranges.
func prefixContainIP(prefix netip.Prefix, ip netip.Addr) bool {
	// if the IP is the network address is not contained
	if prefix.Masked().Addr() == ip {
		return false
	}
	// the broadcast address is not considered contained for IPv4
	if !ip.Is6() {
		ipLast, err := broadcastAddress(prefix)
		if err != nil || ipLast == ip {
			return false
		}
	}
	return prefix.Contains(ip)
}

// TODO(aojea) consolidate all these IPs utils
// pkg/registry/core/service/ipallocator/ipallocator.go
// broadcastAddress returns the broadcast address of the subnet
// The broadcast address is obtained by setting all the host bits
// in a subnet to 1.
// network 192.168.0.0/24 : subnet bits 24 host bits 32 - 24 = 8
// broadcast address 192.168.0.255
func broadcastAddress(subnet netip.Prefix) (netip.Addr, error) {
	base := subnet.Masked().Addr()
	bytes := base.AsSlice()
	// get all the host bits from the subnet
	n := 8*len(bytes) - subnet.Bits()
	// set all the host bits to 1
	for i := len(bytes) - 1; i >= 0 && n > 0; i-- {
		if n >= 8 {
			bytes[i] = 0xff
			n -= 8
		} else {
			mask := ^uint8(0) >> (8 - n)
			bytes[i] |= mask
			break
		}
	}

	addr, ok := netip.AddrFromSlice(bytes)
	if !ok {
		return netip.Addr{}, fmt.Errorf("invalid address %v", bytes)
	}
	return addr, nil
}
