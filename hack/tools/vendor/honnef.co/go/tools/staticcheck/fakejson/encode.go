// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains a modified copy of the encoding/json encoder.
// All dynamic behavior has been removed, and reflecttion has been replaced with go/types.
// This allows us to statically find unmarshable types
// with the same rules for tags, shadowing and addressability as encoding/json.
// This is used for SA1026.

package fakejson

import (
	"go/token"
	"go/types"
	"sort"
	"strings"
	"unicode"

	"honnef.co/go/tools/staticcheck/fakereflect"
)

// parseTag splits a struct field's json tag into its name and
// comma-separated options.
func parseTag(tag string) string {
	if idx := strings.Index(tag, ","); idx != -1 {
		return tag[:idx]
	}
	return tag
}

func Marshal(v types.Type) *UnsupportedTypeError {
	enc := encoder{
		seen: map[fakereflect.TypeAndCanAddr]struct{}{},
	}
	return enc.newTypeEncoder(fakereflect.TypeAndCanAddr{Type: v}, "x")
}

// An UnsupportedTypeError is returned by Marshal when attempting
// to encode an unsupported value type.
type UnsupportedTypeError struct {
	Type types.Type
	Path string
}

var marshalerType = types.NewInterfaceType([]*types.Func{
	types.NewFunc(token.NoPos, nil, "MarshalJSON", types.NewSignature(nil,
		types.NewTuple(),
		types.NewTuple(
			types.NewVar(token.NoPos, nil, "", types.NewSlice(types.Typ[types.Byte])),
			types.NewVar(0, nil, "", types.Universe.Lookup("error").Type())),
		false,
	)),
}, nil).Complete()

var textMarshalerType = types.NewInterfaceType([]*types.Func{
	types.NewFunc(token.NoPos, nil, "MarshalText", types.NewSignature(nil,
		types.NewTuple(),
		types.NewTuple(
			types.NewVar(token.NoPos, nil, "", types.NewSlice(types.Typ[types.Byte])),
			types.NewVar(0, nil, "", types.Universe.Lookup("error").Type())),
		false,
	)),
}, nil).Complete()

type encoder struct {
	seen map[fakereflect.TypeAndCanAddr]struct{}
}

func (enc *encoder) newTypeEncoder(t fakereflect.TypeAndCanAddr, stack string) *UnsupportedTypeError {
	if _, ok := enc.seen[t]; ok {
		return nil
	}
	enc.seen[t] = struct{}{}

	if t.Implements(marshalerType) {
		return nil
	}
	if !t.IsPtr() && t.CanAddr() && fakereflect.PtrTo(t).Implements(marshalerType) {
		return nil
	}
	if t.Implements(textMarshalerType) {
		return nil
	}
	if !t.IsPtr() && t.CanAddr() && fakereflect.PtrTo(t).Implements(textMarshalerType) {
		return nil
	}

	switch t.Type.Underlying().(type) {
	case *types.Basic, *types.Interface:
		return nil
	case *types.Struct:
		return enc.typeFields(t, stack)
	case *types.Map:
		return enc.newMapEncoder(t, stack)
	case *types.Slice:
		return enc.newSliceEncoder(t, stack)
	case *types.Array:
		return enc.newArrayEncoder(t, stack)
	case *types.Pointer:
		// we don't have to express the pointer dereference in the path; x.f is syntactic sugar for (*x).f
		return enc.newTypeEncoder(t.Elem(), stack)
	default:
		return &UnsupportedTypeError{t.Type, stack}
	}
}

func (enc *encoder) newMapEncoder(t fakereflect.TypeAndCanAddr, stack string) *UnsupportedTypeError {
	switch t.Key().Type.Underlying().(type) {
	case *types.Basic:
	default:
		if !t.Key().Implements(textMarshalerType) {
			return &UnsupportedTypeError{
				Type: t.Type,
				Path: stack,
			}
		}
	}
	return enc.newTypeEncoder(t.Elem(), stack+"[k]")
}

func (enc *encoder) newSliceEncoder(t fakereflect.TypeAndCanAddr, stack string) *UnsupportedTypeError {
	// Byte slices get special treatment; arrays don't.
	basic, ok := t.Elem().Type.Underlying().(*types.Basic)
	if ok && basic.Kind() == types.Uint8 {
		p := fakereflect.PtrTo(t.Elem())
		if !p.Implements(marshalerType) && !p.Implements(textMarshalerType) {
			return nil
		}
	}
	return enc.newArrayEncoder(t, stack)
}

func (enc *encoder) newArrayEncoder(t fakereflect.TypeAndCanAddr, stack string) *UnsupportedTypeError {
	return enc.newTypeEncoder(t.Elem(), stack+"[0]")
}

func isValidTag(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		switch {
		case strings.ContainsRune("!#$%&()*+-./:;<=>?@[]^_{|}~ ", c):
			// Backslash and quote chars are reserved, but
			// otherwise any punctuation chars are allowed
			// in a tag name.
		case !unicode.IsLetter(c) && !unicode.IsDigit(c):
			return false
		}
	}
	return true
}

func typeByIndex(t fakereflect.TypeAndCanAddr, index []int) fakereflect.TypeAndCanAddr {
	for _, i := range index {
		if t.IsPtr() {
			t = t.Elem()
		}
		t = t.Field(i).Type
	}
	return t
}

func pathByIndex(t fakereflect.TypeAndCanAddr, index []int) string {
	path := ""
	for _, i := range index {
		if t.IsPtr() {
			t = t.Elem()
		}
		path += "." + t.Field(i).Name
		t = t.Field(i).Type
	}
	return path
}

// A field represents a single field found in a struct.
type field struct {
	name string

	tag   bool
	index []int
	typ   fakereflect.TypeAndCanAddr
}

// byIndex sorts field by index sequence.
type byIndex []field

func (x byIndex) Len() int { return len(x) }

func (x byIndex) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

func (x byIndex) Less(i, j int) bool {
	for k, xik := range x[i].index {
		if k >= len(x[j].index) {
			return false
		}
		if xik != x[j].index[k] {
			return xik < x[j].index[k]
		}
	}
	return len(x[i].index) < len(x[j].index)
}

// typeFields returns a list of fields that JSON should recognize for the given type.
// The algorithm is breadth-first search over the set of structs to include - the top struct
// and then any reachable anonymous structs.
func (enc *encoder) typeFields(t fakereflect.TypeAndCanAddr, stack string) *UnsupportedTypeError {
	// Anonymous fields to explore at the current level and the next.
	current := []field{}
	next := []field{{typ: t}}

	// Count of queued names for current level and the next.
	var count, nextCount map[fakereflect.TypeAndCanAddr]int

	// Types already visited at an earlier level.
	visited := map[fakereflect.TypeAndCanAddr]bool{}

	// Fields found.
	var fields []field

	for len(next) > 0 {
		current, next = next, current[:0]
		count, nextCount = nextCount, map[fakereflect.TypeAndCanAddr]int{}

		for _, f := range current {
			if visited[f.typ] {
				continue
			}
			visited[f.typ] = true

			// Scan f.typ for fields to include.
			for i := 0; i < f.typ.NumField(); i++ {
				sf := f.typ.Field(i)
				if sf.Anonymous {
					t := sf.Type
					if t.IsPtr() {
						t = t.Elem()
					}
					if !sf.IsExported() && !t.IsStruct() {
						// Ignore embedded fields of unexported non-struct types.
						continue
					}
					// Do not ignore embedded fields of unexported struct types
					// since they may have exported fields.
				} else if !sf.IsExported() {
					// Ignore unexported non-embedded fields.
					continue
				}
				tag := sf.Tag.Get("json")
				if tag == "-" {
					continue
				}
				name := parseTag(tag)
				if !isValidTag(name) {
					name = ""
				}
				index := make([]int, len(f.index)+1)
				copy(index, f.index)
				index[len(f.index)] = i

				ft := sf.Type
				if ft.Name() == "" && ft.IsPtr() {
					// Follow pointer.
					ft = ft.Elem()
				}

				// Record found field and index sequence.
				if name != "" || !sf.Anonymous || !ft.IsStruct() {
					tagged := name != ""
					if name == "" {
						name = sf.Name
					}
					field := field{
						name:  name,
						tag:   tagged,
						index: index,
						typ:   ft,
					}

					fields = append(fields, field)
					if count[f.typ] > 1 {
						// If there were multiple instances, add a second,
						// so that the annihilation code will see a duplicate.
						// It only cares about the distinction between 1 or 2,
						// so don't bother generating any more copies.
						fields = append(fields, fields[len(fields)-1])
					}
					continue
				}

				// Record new anonymous struct to explore in next round.
				nextCount[ft]++
				if nextCount[ft] == 1 {
					next = append(next, field{name: ft.Name(), index: index, typ: ft})
				}
			}
		}
	}

	sort.Slice(fields, func(i, j int) bool {
		x := fields
		// sort field by name, breaking ties with depth, then
		// breaking ties with "name came from json tag", then
		// breaking ties with index sequence.
		if x[i].name != x[j].name {
			return x[i].name < x[j].name
		}
		if len(x[i].index) != len(x[j].index) {
			return len(x[i].index) < len(x[j].index)
		}
		if x[i].tag != x[j].tag {
			return x[i].tag
		}
		return byIndex(x).Less(i, j)
	})

	// Delete all fields that are hidden by the Go rules for embedded fields,
	// except that fields with JSON tags are promoted.

	// The fields are sorted in primary order of name, secondary order
	// of field index length. Loop over names; for each name, delete
	// hidden fields by choosing the one dominant field that survives.
	out := fields[:0]
	for advance, i := 0, 0; i < len(fields); i += advance {
		// One iteration per name.
		// Find the sequence of fields with the name of this first field.
		fi := fields[i]
		name := fi.name
		for advance = 1; i+advance < len(fields); advance++ {
			fj := fields[i+advance]
			if fj.name != name {
				break
			}
		}
		if advance == 1 { // Only one field with this name
			out = append(out, fi)
			continue
		}
		dominant, ok := dominantField(fields[i : i+advance])
		if ok {
			out = append(out, dominant)
		}
	}

	fields = out
	sort.Sort(byIndex(fields))

	for i := range fields {
		f := &fields[i]
		err := enc.newTypeEncoder(typeByIndex(t, f.index), stack+pathByIndex(t, f.index))
		if err != nil {
			return err
		}
	}
	return nil
}

// dominantField looks through the fields, all of which are known to
// have the same name, to find the single field that dominates the
// others using Go's embedding rules, modified by the presence of
// JSON tags. If there are multiple top-level fields, the boolean
// will be false: This condition is an error in Go and we skip all
// the fields.
func dominantField(fields []field) (field, bool) {
	// The fields are sorted in increasing index-length order, then by presence of tag.
	// That means that the first field is the dominant one. We need only check
	// for error cases: two fields at top level, either both tagged or neither tagged.
	if len(fields) > 1 && len(fields[0].index) == len(fields[1].index) && fields[0].tag == fields[1].tag {
		return field{}, false
	}
	return fields[0], true
}
