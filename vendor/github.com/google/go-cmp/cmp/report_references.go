// Copyright 2020, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmp

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/google/go-cmp/cmp/internal/flags"
	"github.com/google/go-cmp/cmp/internal/value"
)

const (
	pointerDelimPrefix = "⟪"
	pointerDelimSuffix = "⟫"
)

// formatPointer prints the address of the pointer.
func formatPointer(p value.Pointer, withDelims bool) string {
	v := p.Uintptr()
	if flags.Deterministic {
		v = 0xdeadf00f // Only used for stable testing purposes
	}
	if withDelims {
		return pointerDelimPrefix + formatHex(uint64(v)) + pointerDelimSuffix
	}
	return formatHex(uint64(v))
}

// pointerReferences is a stack of pointers visited so far.
type pointerReferences [][2]value.Pointer

func (ps *pointerReferences) PushPair(vx, vy reflect.Value, d diffMode, deref bool) (pp [2]value.Pointer) {
	if deref && vx.IsValid() {
		vx = vx.Addr()
	}
	if deref && vy.IsValid() {
		vy = vy.Addr()
	}
	switch d {
	case diffUnknown, diffIdentical:
		pp = [2]value.Pointer{value.PointerOf(vx), value.PointerOf(vy)}
	case diffRemoved:
		pp = [2]value.Pointer{value.PointerOf(vx), value.Pointer{}}
	case diffInserted:
		pp = [2]value.Pointer{value.Pointer{}, value.PointerOf(vy)}
	}
	*ps = append(*ps, pp)
	return pp
}

func (ps *pointerReferences) Push(v reflect.Value) (p value.Pointer, seen bool) {
	p = value.PointerOf(v)
	for _, pp := range *ps {
		if p == pp[0] || p == pp[1] {
			return p, true
		}
	}
	*ps = append(*ps, [2]value.Pointer{p, p})
	return p, false
}

func (ps *pointerReferences) Pop() {
	*ps = (*ps)[:len(*ps)-1]
}

// trunkReferences is metadata for a textNode indicating that the sub-tree
// represents the value for either pointer in a pair of references.
type trunkReferences struct{ pp [2]value.Pointer }

// trunkReference is metadata for a textNode indicating that the sub-tree
// represents the value for the given pointer reference.
type trunkReference struct{ p value.Pointer }

// leafReference is metadata for a textNode indicating that the value is
// truncated as it refers to another part of the tree (i.e., a trunk).
type leafReference struct{ p value.Pointer }

func wrapTrunkReferences(pp [2]value.Pointer, s textNode) textNode {
	switch {
	case pp[0].IsNil():
		return &textWrap{Value: s, Metadata: trunkReference{pp[1]}}
	case pp[1].IsNil():
		return &textWrap{Value: s, Metadata: trunkReference{pp[0]}}
	case pp[0] == pp[1]:
		return &textWrap{Value: s, Metadata: trunkReference{pp[0]}}
	default:
		return &textWrap{Value: s, Metadata: trunkReferences{pp}}
	}
}
func wrapTrunkReference(p value.Pointer, printAddress bool, s textNode) textNode {
	var prefix string
	if printAddress {
		prefix = formatPointer(p, true)
	}
	return &textWrap{Prefix: prefix, Value: s, Metadata: trunkReference{p}}
}
func makeLeafReference(p value.Pointer, printAddress bool) textNode {
	out := &textWrap{Prefix: "(", Value: textEllipsis, Suffix: ")"}
	var prefix string
	if printAddress {
		prefix = formatPointer(p, true)
	}
	return &textWrap{Prefix: prefix, Value: out, Metadata: leafReference{p}}
}

// resolveReferences walks the textNode tree searching for any leaf reference
// metadata and resolves each against the corresponding trunk references.
// Since pointer addresses in memory are not particularly readable to the user,
// it replaces each pointer value with an arbitrary and unique reference ID.
func resolveReferences(s textNode) {
	var walkNodes func(textNode, func(textNode))
	walkNodes = func(s textNode, f func(textNode)) {
		f(s)
		switch s := s.(type) {
		case *textWrap:
			walkNodes(s.Value, f)
		case textList:
			for _, r := range s {
				walkNodes(r.Value, f)
			}
		}
	}

	// Collect all trunks and leaves with reference metadata.
	var trunks, leaves []*textWrap
	walkNodes(s, func(s textNode) {
		if s, ok := s.(*textWrap); ok {
			switch s.Metadata.(type) {
			case leafReference:
				leaves = append(leaves, s)
			case trunkReference, trunkReferences:
				trunks = append(trunks, s)
			}
		}
	})

	// No leaf references to resolve.
	if len(leaves) == 0 {
		return
	}

	// Collect the set of all leaf references to resolve.
	leafPtrs := make(map[value.Pointer]bool)
	for _, leaf := range leaves {
		leafPtrs[leaf.Metadata.(leafReference).p] = true
	}

	// Collect the set of trunk pointers that are always paired together.
	// This allows us to assign a single ID to both pointers for brevity.
	// If a pointer in a pair ever occurs by itself or as a different pair,
	// then the pair is broken.
	pairedTrunkPtrs := make(map[value.Pointer]value.Pointer)
	unpair := func(p value.Pointer) {
		if !pairedTrunkPtrs[p].IsNil() {
			pairedTrunkPtrs[pairedTrunkPtrs[p]] = value.Pointer{} // invalidate other half
		}
		pairedTrunkPtrs[p] = value.Pointer{} // invalidate this half
	}
	for _, trunk := range trunks {
		switch p := trunk.Metadata.(type) {
		case trunkReference:
			unpair(p.p) // standalone pointer cannot be part of a pair
		case trunkReferences:
			p0, ok0 := pairedTrunkPtrs[p.pp[0]]
			p1, ok1 := pairedTrunkPtrs[p.pp[1]]
			switch {
			case !ok0 && !ok1:
				// Register the newly seen pair.
				pairedTrunkPtrs[p.pp[0]] = p.pp[1]
				pairedTrunkPtrs[p.pp[1]] = p.pp[0]
			case ok0 && ok1 && p0 == p.pp[1] && p1 == p.pp[0]:
				// Exact pair already seen; do nothing.
			default:
				// Pair conflicts with some other pair; break all pairs.
				unpair(p.pp[0])
				unpair(p.pp[1])
			}
		}
	}

	// Correlate each pointer referenced by leaves to a unique identifier,
	// and print the IDs for each trunk that matches those pointers.
	var nextID uint
	ptrIDs := make(map[value.Pointer]uint)
	newID := func() uint {
		id := nextID
		nextID++
		return id
	}
	for _, trunk := range trunks {
		switch p := trunk.Metadata.(type) {
		case trunkReference:
			if print := leafPtrs[p.p]; print {
				id, ok := ptrIDs[p.p]
				if !ok {
					id = newID()
					ptrIDs[p.p] = id
				}
				trunk.Prefix = updateReferencePrefix(trunk.Prefix, formatReference(id))
			}
		case trunkReferences:
			print0 := leafPtrs[p.pp[0]]
			print1 := leafPtrs[p.pp[1]]
			if print0 || print1 {
				id0, ok0 := ptrIDs[p.pp[0]]
				id1, ok1 := ptrIDs[p.pp[1]]
				isPair := pairedTrunkPtrs[p.pp[0]] == p.pp[1] && pairedTrunkPtrs[p.pp[1]] == p.pp[0]
				if isPair {
					var id uint
					assert(ok0 == ok1) // must be seen together or not at all
					if ok0 {
						assert(id0 == id1) // must have the same ID
						id = id0
					} else {
						id = newID()
						ptrIDs[p.pp[0]] = id
						ptrIDs[p.pp[1]] = id
					}
					trunk.Prefix = updateReferencePrefix(trunk.Prefix, formatReference(id))
				} else {
					if print0 && !ok0 {
						id0 = newID()
						ptrIDs[p.pp[0]] = id0
					}
					if print1 && !ok1 {
						id1 = newID()
						ptrIDs[p.pp[1]] = id1
					}
					switch {
					case print0 && print1:
						trunk.Prefix = updateReferencePrefix(trunk.Prefix, formatReference(id0)+","+formatReference(id1))
					case print0:
						trunk.Prefix = updateReferencePrefix(trunk.Prefix, formatReference(id0))
					case print1:
						trunk.Prefix = updateReferencePrefix(trunk.Prefix, formatReference(id1))
					}
				}
			}
		}
	}

	// Update all leaf references with the unique identifier.
	for _, leaf := range leaves {
		if id, ok := ptrIDs[leaf.Metadata.(leafReference).p]; ok {
			leaf.Prefix = updateReferencePrefix(leaf.Prefix, formatReference(id))
		}
	}
}

func formatReference(id uint) string {
	return fmt.Sprintf("ref#%d", id)
}

func updateReferencePrefix(prefix, ref string) string {
	if prefix == "" {
		return pointerDelimPrefix + ref + pointerDelimSuffix
	}
	suffix := strings.TrimPrefix(prefix, pointerDelimPrefix)
	return pointerDelimPrefix + ref + ": " + suffix
}
