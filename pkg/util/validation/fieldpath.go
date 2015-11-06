/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"bytes"
	"fmt"
	"strconv"
)

// FieldPath represents the path from some root to a particular field.
type FieldPath struct {
	name   string     // the name of this field or "" if this is an index
	index  string     // if name == "", this is a subscript (index or map key) of the previous element
	parent *FieldPath // nil if this is the root element
}

// NewFieldPath creates a root FieldPath object.
func NewFieldPath(name string, moreNames ...string) *FieldPath {
	r := &FieldPath{name: name, parent: nil}
	for _, anotherName := range moreNames {
		r = &FieldPath{name: anotherName, parent: r}
	}
	return r
}

// Root returns the root element of this FieldPath.
func (fp *FieldPath) Root() *FieldPath {
	for ; fp.parent != nil; fp = fp.parent {
		// Do nothing.
	}
	return fp
}

// Child creates a new FieldPath that is a child of the method receiver.
func (fp *FieldPath) Child(name string, moreNames ...string) *FieldPath {
	r := NewFieldPath(name, moreNames...)
	r.Root().parent = fp
	return r
}

// Index indicates that the previous FieldPath is to be subscripted by an int.
// This sets the same underlying value as Key.
func (fp *FieldPath) Index(index int) *FieldPath {
	return &FieldPath{index: strconv.Itoa(index), parent: fp}
}

// Key indicates that the previous FieldPath is to be subscripted by a string.
// This sets the same underlying value as Index.
func (fp *FieldPath) Key(key string) *FieldPath {
	return &FieldPath{index: key, parent: fp}
}

// String produces a string representation of the FieldPath.
func (fp *FieldPath) String() string {
	// make a slice to iterate
	elems := []*FieldPath{}
	for p := fp; p != nil; p = p.parent {
		elems = append(elems, p)
	}

	// iterate, but it has to be backwards
	buf := bytes.NewBuffer(nil)
	for i := range elems {
		p := elems[len(elems)-1-i]
		if p.parent != nil && len(p.name) > 0 {
			// This is either the root or it is a subscript.
			buf.WriteString(".")
		}
		if len(p.name) > 0 {
			buf.WriteString(p.name)
		} else {
			fmt.Fprintf(buf, "[%s]", p.index)
		}
	}
	return buf.String()
}
