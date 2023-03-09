// Copyright 2017 The Bazel Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package starlarkstruct defines the Starlark types 'struct' and
// 'module', both optional language extensions.
//
package starlarkstruct // import "go.starlark.net/starlarkstruct"

// It is tempting to introduce a variant of Struct that is a wrapper
// around a Go struct value, for stronger typing guarantees and more
// efficient and convenient field lookup. However:
// 1) all fields of Starlark structs are optional, so we cannot represent
//    them using more specific types such as String, Int, *Depset, and
//    *File, as such types give no way to represent missing fields.
// 2) the efficiency gain of direct struct field access is rather
//    marginal: finding the index of a field by binary searching on the
//    sorted list of field names is quite fast compared to the other
//    overheads.
// 3) the gains in compactness and spatial locality are also rather
//    marginal: the array behind the []entry slice is (due to field name
//    strings) only a factor of 2 larger than the corresponding Go struct
//    would be, and, like the Go struct, requires only a single allocation.

import (
	"fmt"
	"sort"
	"strings"

	"go.starlark.net/starlark"
	"go.starlark.net/syntax"
)

// Make is the implementation of a built-in function that instantiates
// an immutable struct from the specified keyword arguments.
//
// An application can add 'struct' to the Starlark environment like so:
//
// 	globals := starlark.StringDict{
// 		"struct":  starlark.NewBuiltin("struct", starlarkstruct.Make),
// 	}
//
func Make(_ *starlark.Thread, _ *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	if len(args) > 0 {
		return nil, fmt.Errorf("struct: unexpected positional arguments")
	}
	return FromKeywords(Default, kwargs), nil
}

// FromKeywords returns a new struct instance whose fields are specified by the
// key/value pairs in kwargs.  (Each kwargs[i][0] must be a starlark.String.)
func FromKeywords(constructor starlark.Value, kwargs []starlark.Tuple) *Struct {
	if constructor == nil {
		panic("nil constructor")
	}
	s := &Struct{
		constructor: constructor,
		entries:     make(entries, 0, len(kwargs)),
	}
	for _, kwarg := range kwargs {
		k := string(kwarg[0].(starlark.String))
		v := kwarg[1]
		s.entries = append(s.entries, entry{k, v})
	}
	sort.Sort(s.entries)
	return s
}

// FromStringDict returns a new struct instance whose elements are those of d.
// The constructor parameter specifies the constructor; use Default for an ordinary struct.
func FromStringDict(constructor starlark.Value, d starlark.StringDict) *Struct {
	if constructor == nil {
		panic("nil constructor")
	}
	s := &Struct{
		constructor: constructor,
		entries:     make(entries, 0, len(d)),
	}
	for k, v := range d {
		s.entries = append(s.entries, entry{k, v})
	}
	sort.Sort(s.entries)
	return s
}

// Struct is an immutable Starlark type that maps field names to values.
// It is not iterable and does not support len.
//
// A struct has a constructor, a distinct value that identifies a class
// of structs, and which appears in the struct's string representation.
//
// Operations such as x+y fail if the constructors of the two operands
// are not equal.
//
// The default constructor, Default, is the string "struct", but
// clients may wish to 'brand' structs for their own purposes.
// The constructor value appears in the printed form of the value,
// and is accessible using the Constructor method.
//
// Use Attr to access its fields and AttrNames to enumerate them.
type Struct struct {
	constructor starlark.Value
	entries     entries // sorted by name
}

// Default is the default constructor for structs.
// It is merely the string "struct".
const Default = starlark.String("struct")

type entries []entry

func (a entries) Len() int           { return len(a) }
func (a entries) Less(i, j int) bool { return a[i].name < a[j].name }
func (a entries) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

type entry struct {
	name  string
	value starlark.Value
}

var (
	_ starlark.HasAttrs  = (*Struct)(nil)
	_ starlark.HasBinary = (*Struct)(nil)
)

// ToStringDict adds a name/value entry to d for each field of the struct.
func (s *Struct) ToStringDict(d starlark.StringDict) {
	for _, e := range s.entries {
		d[e.name] = e.value
	}
}

func (s *Struct) String() string {
	buf := new(strings.Builder)
	if s.constructor == Default {
		// NB: The Java implementation always prints struct
		// even for Bazel provider instances.
		buf.WriteString("struct") // avoid String()'s quotation
	} else {
		buf.WriteString(s.constructor.String())
	}
	buf.WriteByte('(')
	for i, e := range s.entries {
		if i > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(e.name)
		buf.WriteString(" = ")
		buf.WriteString(e.value.String())
	}
	buf.WriteByte(')')
	return buf.String()
}

// Constructor returns the constructor used to create this struct.
func (s *Struct) Constructor() starlark.Value { return s.constructor }

func (s *Struct) Type() string         { return "struct" }
func (s *Struct) Truth() starlark.Bool { return true } // even when empty
func (s *Struct) Hash() (uint32, error) {
	// Same algorithm as Tuple.hash, but with different primes.
	var x, m uint32 = 8731, 9839
	for _, e := range s.entries {
		namehash, _ := starlark.String(e.name).Hash()
		x = x ^ 3*namehash
		y, err := e.value.Hash()
		if err != nil {
			return 0, err
		}
		x = x ^ y*m
		m += 7349
	}
	return x, nil
}
func (s *Struct) Freeze() {
	for _, e := range s.entries {
		e.value.Freeze()
	}
}

func (x *Struct) Binary(op syntax.Token, y starlark.Value, side starlark.Side) (starlark.Value, error) {
	if y, ok := y.(*Struct); ok && op == syntax.PLUS {
		if side == starlark.Right {
			x, y = y, x
		}

		if eq, err := starlark.Equal(x.constructor, y.constructor); err != nil {
			return nil, fmt.Errorf("in %s + %s: error comparing constructors: %v",
				x.constructor, y.constructor, err)
		} else if !eq {
			return nil, fmt.Errorf("cannot add structs of different constructors: %s + %s",
				x.constructor, y.constructor)
		}

		z := make(starlark.StringDict, x.len()+y.len())
		for _, e := range x.entries {
			z[e.name] = e.value
		}
		for _, e := range y.entries {
			z[e.name] = e.value
		}

		return FromStringDict(x.constructor, z), nil
	}
	return nil, nil // unhandled
}

// Attr returns the value of the specified field.
func (s *Struct) Attr(name string) (starlark.Value, error) {
	// Binary search the entries.
	// This implementation is a specialization of
	// sort.Search that avoids dynamic dispatch.
	n := len(s.entries)
	i, j := 0, n
	for i < j {
		h := int(uint(i+j) >> 1)
		if s.entries[h].name < name {
			i = h + 1
		} else {
			j = h
		}
	}
	if i < n && s.entries[i].name == name {
		return s.entries[i].value, nil
	}

	var ctor string
	if s.constructor != Default {
		ctor = s.constructor.String() + " "
	}
	return nil, starlark.NoSuchAttrError(
		fmt.Sprintf("%sstruct has no .%s attribute", ctor, name))
}

func (s *Struct) len() int { return len(s.entries) }

// AttrNames returns a new sorted list of the struct fields.
func (s *Struct) AttrNames() []string {
	names := make([]string, len(s.entries))
	for i, e := range s.entries {
		names[i] = e.name
	}
	return names
}

func (x *Struct) CompareSameType(op syntax.Token, y_ starlark.Value, depth int) (bool, error) {
	y := y_.(*Struct)
	switch op {
	case syntax.EQL:
		return structsEqual(x, y, depth)
	case syntax.NEQ:
		eq, err := structsEqual(x, y, depth)
		return !eq, err
	default:
		return false, fmt.Errorf("%s %s %s not implemented", x.Type(), op, y.Type())
	}
}

func structsEqual(x, y *Struct, depth int) (bool, error) {
	if x.len() != y.len() {
		return false, nil
	}

	if eq, err := starlark.Equal(x.constructor, y.constructor); err != nil {
		return false, fmt.Errorf("error comparing struct constructors %v and %v: %v",
			x.constructor, y.constructor, err)
	} else if !eq {
		return false, nil
	}

	for i, n := 0, x.len(); i < n; i++ {
		if x.entries[i].name != y.entries[i].name {
			return false, nil
		} else if eq, err := starlark.EqualDepth(x.entries[i].value, y.entries[i].value, depth-1); err != nil {
			return false, err
		} else if !eq {
			return false, nil
		}
	}
	return true, nil
}
