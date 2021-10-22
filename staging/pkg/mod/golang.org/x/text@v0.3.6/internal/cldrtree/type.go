// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cldrtree

import (
	"log"
	"strconv"
)

// enumIndex is the numerical value of an enum value.
type enumIndex int

// An enum is a collection of enum values.
type enum struct {
	name   string // the Go type of the enum
	rename func(string) string
	keyMap map[string]enumIndex
	keys   []string
}

// lookup returns the index for the enum corresponding to the string. If s
// currently does not exist it will add the entry.
func (e *enum) lookup(s string) enumIndex {
	if e.rename != nil {
		s = e.rename(s)
	}
	x, ok := e.keyMap[s]
	if !ok {
		if e.keyMap == nil {
			e.keyMap = map[string]enumIndex{}
		}
		u, err := strconv.ParseUint(s, 10, 32)
		if err == nil {
			for len(e.keys) <= int(u) {
				x := enumIndex(len(e.keys))
				s := strconv.Itoa(int(x))
				e.keyMap[s] = x
				e.keys = append(e.keys, s)
			}
			if e.keyMap[s] != enumIndex(u) {
				// TODO: handle more gracefully.
				log.Fatalf("cldrtree: mix of integer and non-integer for %q %v", s, e.keys)
			}
			return enumIndex(u)
		}
		x = enumIndex(len(e.keys))
		e.keyMap[s] = x
		e.keys = append(e.keys, s)
	}
	return x
}

// A typeInfo indicates the set of possible enum values and a mapping from
// these values to subtypes.
type typeInfo struct {
	enum        *enum
	entries     map[enumIndex]*typeInfo
	keyTypeInfo *typeInfo
	shareKeys   bool
}

func (t *typeInfo) sharedKeys() bool {
	return t.shareKeys
}

func (t *typeInfo) lookupSubtype(s string, opts *options) (x enumIndex, sub *typeInfo) {
	if t.enum == nil {
		if t.enum = opts.sharedEnums; t.enum == nil {
			t.enum = &enum{}
		}
	}
	if opts.sharedEnums != nil && t.enum != opts.sharedEnums {
		panic("incompatible enums defined")
	}
	x = t.enum.lookup(s)
	if t.entries == nil {
		t.entries = map[enumIndex]*typeInfo{}
	}
	sub, ok := t.entries[x]
	if !ok {
		sub = opts.sharedType
		if sub == nil {
			sub = &typeInfo{}
		}
		t.entries[x] = sub
	}
	t.shareKeys = opts.sharedType != nil // For analysis purposes.
	return x, sub
}

// metaData includes information about subtypes, possibly sharing commonality
// with sibling branches, and information about inheritance, which may differ
// per branch.
type metaData struct {
	b *Builder

	parent *metaData

	index    enumIndex // index into the parent's subtype index
	key      string
	elem     string // XML element corresponding to this type.
	typeInfo *typeInfo

	lookup map[enumIndex]*metaData
	subs   []*metaData

	inheritOffset int    // always negative when applicable
	inheritIndex  string // new value for field indicated by inheritOffset
	// inheritType   *metaData
}

func (m *metaData) sub(key string, opts *options) *metaData {
	if m.lookup == nil {
		m.lookup = map[enumIndex]*metaData{}
	}
	enum, info := m.typeInfo.lookupSubtype(key, opts)
	sub := m.lookup[enum]
	if sub == nil {
		sub = &metaData{
			b:      m.b,
			parent: m,

			index:    enum,
			key:      key,
			typeInfo: info,
		}
		m.lookup[enum] = sub
		m.subs = append(m.subs, sub)
	}
	return sub
}

func (m *metaData) validate() {
	for _, s := range m.subs {
		s.validate()
	}
}
