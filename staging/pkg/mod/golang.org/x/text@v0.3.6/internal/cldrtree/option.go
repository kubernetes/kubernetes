// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cldrtree

import (
	"reflect"

	"golang.org/x/text/unicode/cldr"
)

// An Option configures an Index.
type Option func(*options)

type options struct {
	parent *Index

	name  string
	alias *cldr.Common

	sharedType  *typeInfo
	sharedEnums *enum
}

func (o *options) fill(opt []Option) {
	for _, f := range opt {
		f(o)
	}
}

// aliasOpt sets an alias from the given node, if the node defines one.
func (o *options) setAlias(n Element) {
	if n != nil && !reflect.ValueOf(n).IsNil() {
		o.alias = n.GetCommon()
	}
}

// Enum defines a enumeration type. The resulting option may be passed for the
// construction of multiple Indexes, which they will share the same enum values.
// Calling Gen on a Builder will generate the Enum for the given name. The
// optional values fix the values for the given identifier to the argument
// position (starting at 0). Other values may still be added and will be
// assigned to subsequent values.
func Enum(name string, value ...string) Option {
	return EnumFunc(name, nil, value...)
}

// EnumFunc is like Enum but also takes a function that allows rewriting keys.
func EnumFunc(name string, rename func(string) string, value ...string) Option {
	enum := &enum{name: name, rename: rename, keyMap: map[string]enumIndex{}}
	for _, e := range value {
		enum.lookup(e)
	}
	return func(o *options) {
		found := false
		for _, e := range o.parent.meta.b.enums {
			if e.name == enum.name {
				found = true
				break
			}
		}
		if !found {
			o.parent.meta.b.enums = append(o.parent.meta.b.enums, enum)
		}
		o.sharedEnums = enum
	}
}

// SharedType returns an option which causes all Indexes to which this option is
// passed to have the same type.
func SharedType() Option {
	info := &typeInfo{}
	return func(o *options) { o.sharedType = info }
}

func useSharedType() Option {
	return func(o *options) {
		sub := o.parent.meta.typeInfo.keyTypeInfo
		if sub == nil {
			sub = &typeInfo{}
			o.parent.meta.typeInfo.keyTypeInfo = sub
		}
		o.sharedType = sub
	}
}
