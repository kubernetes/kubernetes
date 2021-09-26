// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cldrtree

import (
	"golang.org/x/text/internal/language/compact"
	"golang.org/x/text/language"
)

const (
	inheritOffsetShift        = 12
	inheritMask        uint16 = 0x8000
	inheritValueMask   uint16 = 0x0FFF

	missingValue uint16 = 0xFFFF
)

// Tree holds a tree of CLDR data.
type Tree struct {
	Locales []uint32
	Indices []uint16
	Buckets []string
}

// Lookup looks up CLDR data for the given path. The lookup adheres to the alias
// and locale inheritance rules as defined in CLDR.
//
// Each subsequent element in path indicates which subtree to select data from.
// The last element of the path must select a leaf node. All other elements
// of the path select a subindex.
func (t *Tree) Lookup(tag compact.ID, path ...uint16) string {
	return t.lookup(tag, false, path...)
}

// LookupFeature is like Lookup, but will first check whether a value of "other"
// as a fallback before traversing the inheritance chain.
func (t *Tree) LookupFeature(tag compact.ID, path ...uint16) string {
	return t.lookup(tag, true, path...)
}

func (t *Tree) lookup(tag compact.ID, isFeature bool, path ...uint16) string {
	origLang := tag
outer:
	for {
		index := t.Indices[t.Locales[tag]:]

		k := uint16(0)
		for i := range path {
			max := index[k]
			if i < len(path)-1 {
				// index (non-leaf)
				if path[i] >= max {
					break
				}
				k = index[k+1+path[i]]
				if k == 0 {
					break
				}
				if v := k &^ inheritMask; k != v {
					offset := v >> inheritOffsetShift
					value := v & inheritValueMask
					path[uint16(i)-offset] = value
					tag = origLang
					continue outer
				}
			} else {
				// leaf value
				offset := missingValue
				if path[i] < max {
					offset = index[k+2+path[i]]
				}
				if offset == missingValue {
					if !isFeature {
						break
					}
					// "other" feature must exist
					offset = index[k+2]
				}
				data := t.Buckets[index[k+1]]
				n := uint16(data[offset])
				return data[offset+1 : offset+n+1]
			}
		}
		if tag == 0 {
			break
		}
		tag = tag.Parent()
	}
	return ""
}

func build(b *Builder) (*Tree, error) {
	var t Tree

	t.Locales = make([]uint32, language.NumCompactTags)

	for _, loc := range b.locales {
		tag, _ := language.CompactIndex(loc.tag)
		t.Locales[tag] = uint32(len(t.Indices))
		var x indexBuilder
		x.add(loc.root)
		t.Indices = append(t.Indices, x.index...)
	}
	// Set locales for which we don't have data to the parent's data.
	for i, v := range t.Locales {
		p := compact.ID(i)
		for v == 0 && p != 0 {
			p = p.Parent()
			v = t.Locales[p]
		}
		t.Locales[i] = v
	}

	for _, b := range b.buckets {
		t.Buckets = append(t.Buckets, string(b))
	}
	if b.err != nil {
		return nil, b.err
	}
	return &t, nil
}

type indexBuilder struct {
	index []uint16
}

func (b *indexBuilder) add(i *Index) uint16 {
	offset := len(b.index)

	max := enumIndex(0)
	switch {
	case len(i.values) > 0:
		for _, v := range i.values {
			if v.key > max {
				max = v.key
			}
		}
		b.index = append(b.index, make([]uint16, max+3)...)

		b.index[offset] = uint16(max) + 1

		b.index[offset+1] = i.values[0].value.bucket
		for i := offset + 2; i < len(b.index); i++ {
			b.index[i] = missingValue
		}
		for _, v := range i.values {
			b.index[offset+2+int(v.key)] = v.value.bucketPos
		}
		return uint16(offset)

	case len(i.subIndex) > 0:
		for _, s := range i.subIndex {
			if s.meta.index > max {
				max = s.meta.index
			}
		}
		b.index = append(b.index, make([]uint16, max+2)...)

		b.index[offset] = uint16(max) + 1

		for _, s := range i.subIndex {
			x := b.add(s)
			b.index[offset+int(s.meta.index)+1] = x
		}
		return uint16(offset)

	case i.meta.inheritOffset < 0:
		v := uint16(-(i.meta.inheritOffset + 1)) << inheritOffsetShift
		p := i.meta
		for k := i.meta.inheritOffset; k < 0; k++ {
			p = p.parent
		}
		v += uint16(p.typeInfo.enum.lookup(i.meta.inheritIndex))
		v |= inheritMask
		return v
	}

	return 0
}
