/*
Copyright 2015 The Kubernetes Authors.

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

package field

import (
	"bytes"
	"fmt"
	"strconv"
)

type pathOptions struct {
	path *Path
}

// PathOption modifies a pathOptions
type PathOption func(o *pathOptions)

// WithPath generates a PathOption
func WithPath(p *Path) PathOption {
	return func(o *pathOptions) {
		o.path = p
	}
}

// ToPath produces *Path from a set of PathOption
func ToPath(opts ...PathOption) *Path {
	c := &pathOptions{}
	for _, opt := range opts {
		opt(c)
	}
	return c.path
}

// Path represents the path from some root to a particular field.
type Path struct {
	name   string // the name of this field or "" if this is an index
	index  string // if name == "", this is a subscript (index or map key) of the previous element
	parent *Path  // nil if this is the root element
}

// NewPath creates a root Path object.
func NewPath(name string, moreNames ...string) *Path {
	r := &Path{name: name, parent: nil}
	for _, anotherName := range moreNames {
		r = &Path{name: anotherName, parent: r}
	}
	return r
}

// Root returns the root element of this Path.
func (p *Path) Root() *Path {
	for ; p.parent != nil; p = p.parent {
		// Do nothing.
	}
	return p
}

// Child creates a new Path that is a child of the method receiver.
func (p *Path) Child(name string, moreNames ...string) *Path {
	r := NewPath(name, moreNames...)
	r.Root().parent = p
	return r
}

// Index indicates that the previous Path is to be subscripted by an int.
// This sets the same underlying value as Key.
func (p *Path) Index(index int) *Path {
	return &Path{index: strconv.Itoa(index), parent: p}
}

// Key indicates that the previous Path is to be subscripted by a string.
// This sets the same underlying value as Index.
func (p *Path) Key(key string) *Path {
	return &Path{index: key, parent: p}
}

// String produces a string representation of the Path.
func (p *Path) String() string {
	if p == nil {
		return "<nil>"
	}
	// make a slice to iterate
	elems := []*Path{}
	for ; p != nil; p = p.parent {
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
