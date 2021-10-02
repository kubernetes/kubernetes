// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmpopts

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/google/go-cmp/cmp"
)

// filterField returns a new Option where opt is only evaluated on paths that
// include a specific exported field on a single struct type.
// The struct type is specified by passing in a value of that type.
//
// The name may be a dot-delimited string (e.g., "Foo.Bar") to select a
// specific sub-field that is embedded or nested within the parent struct.
func filterField(typ interface{}, name string, opt cmp.Option) cmp.Option {
	// TODO: This is currently unexported over concerns of how helper filters
	// can be composed together easily.
	// TODO: Add tests for FilterField.

	sf := newStructFilter(typ, name)
	return cmp.FilterPath(sf.filter, opt)
}

type structFilter struct {
	t  reflect.Type // The root struct type to match on
	ft fieldTree    // Tree of fields to match on
}

func newStructFilter(typ interface{}, names ...string) structFilter {
	// TODO: Perhaps allow * as a special identifier to allow ignoring any
	// number of path steps until the next field match?
	// This could be useful when a concrete struct gets transformed into
	// an anonymous struct where it is not possible to specify that by type,
	// but the transformer happens to provide guarantees about the names of
	// the transformed fields.

	t := reflect.TypeOf(typ)
	if t == nil || t.Kind() != reflect.Struct {
		panic(fmt.Sprintf("%T must be a non-pointer struct", typ))
	}
	var ft fieldTree
	for _, name := range names {
		cname, err := canonicalName(t, name)
		if err != nil {
			panic(fmt.Sprintf("%s: %v", strings.Join(cname, "."), err))
		}
		ft.insert(cname)
	}
	return structFilter{t, ft}
}

func (sf structFilter) filter(p cmp.Path) bool {
	for i, ps := range p {
		if ps.Type().AssignableTo(sf.t) && sf.ft.matchPrefix(p[i+1:]) {
			return true
		}
	}
	return false
}

// fieldTree represents a set of dot-separated identifiers.
//
// For example, inserting the following selectors:
//	Foo
//	Foo.Bar.Baz
//	Foo.Buzz
//	Nuka.Cola.Quantum
//
// Results in a tree of the form:
//	{sub: {
//		"Foo": {ok: true, sub: {
//			"Bar": {sub: {
//				"Baz": {ok: true},
//			}},
//			"Buzz": {ok: true},
//		}},
//		"Nuka": {sub: {
//			"Cola": {sub: {
//				"Quantum": {ok: true},
//			}},
//		}},
//	}}
type fieldTree struct {
	ok  bool                 // Whether this is a specified node
	sub map[string]fieldTree // The sub-tree of fields under this node
}

// insert inserts a sequence of field accesses into the tree.
func (ft *fieldTree) insert(cname []string) {
	if ft.sub == nil {
		ft.sub = make(map[string]fieldTree)
	}
	if len(cname) == 0 {
		ft.ok = true
		return
	}
	sub := ft.sub[cname[0]]
	sub.insert(cname[1:])
	ft.sub[cname[0]] = sub
}

// matchPrefix reports whether any selector in the fieldTree matches
// the start of path p.
func (ft fieldTree) matchPrefix(p cmp.Path) bool {
	for _, ps := range p {
		switch ps := ps.(type) {
		case cmp.StructField:
			ft = ft.sub[ps.Name()]
			if ft.ok {
				return true
			}
			if len(ft.sub) == 0 {
				return false
			}
		case cmp.Indirect:
		default:
			return false
		}
	}
	return false
}

// canonicalName returns a list of identifiers where any struct field access
// through an embedded field is expanded to include the names of the embedded
// types themselves.
//
// For example, suppose field "Foo" is not directly in the parent struct,
// but actually from an embedded struct of type "Bar". Then, the canonical name
// of "Foo" is actually "Bar.Foo".
//
// Suppose field "Foo" is not directly in the parent struct, but actually
// a field in two different embedded structs of types "Bar" and "Baz".
// Then the selector "Foo" causes a panic since it is ambiguous which one it
// refers to. The user must specify either "Bar.Foo" or "Baz.Foo".
func canonicalName(t reflect.Type, sel string) ([]string, error) {
	var name string
	sel = strings.TrimPrefix(sel, ".")
	if sel == "" {
		return nil, fmt.Errorf("name must not be empty")
	}
	if i := strings.IndexByte(sel, '.'); i < 0 {
		name, sel = sel, ""
	} else {
		name, sel = sel[:i], sel[i:]
	}

	// Type must be a struct or pointer to struct.
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("%v must be a struct", t)
	}

	// Find the canonical name for this current field name.
	// If the field exists in an embedded struct, then it will be expanded.
	sf, _ := t.FieldByName(name)
	if !isExported(name) {
		// Avoid using reflect.Type.FieldByName for unexported fields due to
		// buggy behavior with regard to embeddeding and unexported fields.
		// See https://golang.org/issue/4876 for details.
		sf = reflect.StructField{}
		for i := 0; i < t.NumField() && sf.Name == ""; i++ {
			if t.Field(i).Name == name {
				sf = t.Field(i)
			}
		}
	}
	if sf.Name == "" {
		return []string{name}, fmt.Errorf("does not exist")
	}
	var ss []string
	for i := range sf.Index {
		ss = append(ss, t.FieldByIndex(sf.Index[:i+1]).Name)
	}
	if sel == "" {
		return ss, nil
	}
	ssPost, err := canonicalName(sf.Type, sel)
	return append(ss, ssPost...), err
}
