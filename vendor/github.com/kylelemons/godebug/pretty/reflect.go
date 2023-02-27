// Copyright 2013 Google Inc.  All rights reserved.
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

package pretty

import (
	"encoding"
	"fmt"
	"reflect"
	"sort"
)

func isZeroVal(val reflect.Value) bool {
	if !val.CanInterface() {
		return false
	}
	z := reflect.Zero(val.Type()).Interface()
	return reflect.DeepEqual(val.Interface(), z)
}

// pointerTracker is a helper for tracking pointer chasing to detect cycles.
type pointerTracker struct {
	addrs map[uintptr]int // addr[address] = seen count

	lastID int
	ids    map[uintptr]int // ids[address] = id
}

// track tracks following a reference (pointer, slice, map, etc).  Every call to
// track should be paired with a call to untrack.
func (p *pointerTracker) track(ptr uintptr) {
	if p.addrs == nil {
		p.addrs = make(map[uintptr]int)
	}
	p.addrs[ptr]++
}

// untrack registers that we have backtracked over the reference to the pointer.
func (p *pointerTracker) untrack(ptr uintptr) {
	p.addrs[ptr]--
	if p.addrs[ptr] == 0 {
		delete(p.addrs, ptr)
	}
}

// seen returns whether the pointer was previously seen along this path.
func (p *pointerTracker) seen(ptr uintptr) bool {
	_, ok := p.addrs[ptr]
	return ok
}

// keep allocates an ID for the given address and returns it.
func (p *pointerTracker) keep(ptr uintptr) int {
	if p.ids == nil {
		p.ids = make(map[uintptr]int)
	}
	if _, ok := p.ids[ptr]; !ok {
		p.lastID++
		p.ids[ptr] = p.lastID
	}
	return p.ids[ptr]
}

// id returns the ID for the given address.
func (p *pointerTracker) id(ptr uintptr) (int, bool) {
	if p.ids == nil {
		p.ids = make(map[uintptr]int)
	}
	id, ok := p.ids[ptr]
	return id, ok
}

// reflector adds local state to the recursive reflection logic.
type reflector struct {
	*Config
	*pointerTracker
}

// follow handles following a possiblly-recursive reference to the given value
// from the given ptr address.
func (r *reflector) follow(ptr uintptr, val reflect.Value) node {
	if r.pointerTracker == nil {
		// Tracking disabled
		return r.val2node(val)
	}

	// If a parent already followed this, emit a reference marker
	if r.seen(ptr) {
		id := r.keep(ptr)
		return ref{id}
	}

	// Track the pointer we're following while on this recursive branch
	r.track(ptr)
	defer r.untrack(ptr)
	n := r.val2node(val)

	// If the recursion used this ptr, wrap it with a target marker
	if id, ok := r.id(ptr); ok {
		return target{id, n}
	}

	// Otherwise, return the node unadulterated
	return n
}

func (r *reflector) val2node(val reflect.Value) node {
	if !val.IsValid() {
		return rawVal("nil")
	}

	if val.CanInterface() {
		v := val.Interface()
		if formatter, ok := r.Formatter[val.Type()]; ok {
			if formatter != nil {
				res := reflect.ValueOf(formatter).Call([]reflect.Value{val})
				return rawVal(res[0].Interface().(string))
			}
		} else {
			if s, ok := v.(fmt.Stringer); ok && r.PrintStringers {
				return stringVal(s.String())
			}
			if t, ok := v.(encoding.TextMarshaler); ok && r.PrintTextMarshalers {
				if raw, err := t.MarshalText(); err == nil { // if NOT an error
					return stringVal(string(raw))
				}
			}
		}
	}

	switch kind := val.Kind(); kind {
	case reflect.Ptr:
		if val.IsNil() {
			return rawVal("nil")
		}
		return r.follow(val.Pointer(), val.Elem())
	case reflect.Interface:
		if val.IsNil() {
			return rawVal("nil")
		}
		return r.val2node(val.Elem())
	case reflect.String:
		return stringVal(val.String())
	case reflect.Slice:
		n := list{}
		length := val.Len()
		ptr := val.Pointer()
		for i := 0; i < length; i++ {
			n = append(n, r.follow(ptr, val.Index(i)))
		}
		return n
	case reflect.Array:
		n := list{}
		length := val.Len()
		for i := 0; i < length; i++ {
			n = append(n, r.val2node(val.Index(i)))
		}
		return n
	case reflect.Map:
		// Extract the keys and sort them for stable iteration
		keys := val.MapKeys()
		pairs := make([]mapPair, 0, len(keys))
		for _, key := range keys {
			pairs = append(pairs, mapPair{
				key:   new(formatter).compactString(r.val2node(key)), // can't be cyclic
				value: val.MapIndex(key),
			})
		}
		sort.Sort(byKey(pairs))

		// Process the keys into the final representation
		ptr, n := val.Pointer(), keyvals{}
		for _, pair := range pairs {
			n = append(n, keyval{
				key: pair.key,
				val: r.follow(ptr, pair.value),
			})
		}
		return n
	case reflect.Struct:
		n := keyvals{}
		typ := val.Type()
		fields := typ.NumField()
		for i := 0; i < fields; i++ {
			sf := typ.Field(i)
			if !r.IncludeUnexported && sf.PkgPath != "" {
				continue
			}
			field := val.Field(i)
			if r.SkipZeroFields && isZeroVal(field) {
				continue
			}
			n = append(n, keyval{sf.Name, r.val2node(field)})
		}
		return n
	case reflect.Bool:
		if val.Bool() {
			return rawVal("true")
		}
		return rawVal("false")
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rawVal(fmt.Sprintf("%d", val.Int()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return rawVal(fmt.Sprintf("%d", val.Uint()))
	case reflect.Uintptr:
		return rawVal(fmt.Sprintf("0x%X", val.Uint()))
	case reflect.Float32, reflect.Float64:
		return rawVal(fmt.Sprintf("%v", val.Float()))
	case reflect.Complex64, reflect.Complex128:
		return rawVal(fmt.Sprintf("%v", val.Complex()))
	}

	// Fall back to the default %#v if we can
	if val.CanInterface() {
		return rawVal(fmt.Sprintf("%#v", val.Interface()))
	}

	return rawVal(val.String())
}

type mapPair struct {
	key   string
	value reflect.Value
}

type byKey []mapPair

func (v byKey) Len() int           { return len(v) }
func (v byKey) Swap(i, j int)      { v[i], v[j] = v[j], v[i] }
func (v byKey) Less(i, j int) bool { return v[i].key < v[j].key }
