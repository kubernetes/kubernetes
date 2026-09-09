/*
Copyright 2025 The Kubernetes Authors.

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

package codetags

import (
	"strconv"
	"strings"
)

// Tag represents a single comment tag with typed args.
type Tag struct {
	// Name is the name of the tag with no arguments.
	Name string

	// Args is a list of optional arguments to the tag.
	Args []Arg

	// Value is the string representation of the tag value.
	// Provides the tag value when ValueType is ValueTypeString, ValueTypeBool, ValueTypeInt or ValueTypeRaw.
	Value string

	// ValueTag is another tag parsed from the value of this tag.
	// Provides the tag value when ValueType is ValueTypeTag.
	ValueTag *Tag

	// ValueType is the type of the value.
	ValueType ValueType
}

// PositionalArg returns the positional argument. If there is no positional
// argument, it returns false.
func (t Tag) PositionalArg() (Arg, bool) {
	if len(t.Args) == 0 || len(t.Args[0].Name) > 0 {
		return Arg{}, false
	}
	return t.Args[0], true
}

// NamedArg returns the named argument. If o named argument is found, it returns
// false. Always returns false for empty name; use PositionalArg instead.
func (t Tag) NamedArg(name string) (Arg, bool) {
	if len(name) == 0 {
		return Arg{}, false
	}
	for _, arg := range t.Args {
		if arg.Name == name {
			return arg, true
		}
	}
	return Arg{}, false
}

// String returns the canonical string representation of the tag.
// All strings are represented in double quotes. Spacing is normalized.
func (t Tag) String() string {
	buf := strings.Builder{}
	buf.WriteString(t.Name)
	if len(t.Args) > 0 {
		buf.WriteString("(")
		for i, a := range t.Args {
			if i > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(a.String())
		}
		buf.WriteString(")")
	}
	if t.ValueType != ValueTypeNone {
		if t.ValueType == ValueTypeTag {
			buf.WriteString("=+")
			buf.WriteString(t.ValueTag.String())
		} else {
			buf.WriteString("=")
			if t.ValueType == ValueTypeString {
				buf.WriteString(strconv.Quote(t.Value))
			} else {
				buf.WriteString(t.Value)
			}
		}
	}
	return buf.String()
}

// Arg represents a argument.
type Arg struct {
	// Name is the name of a named argument. This is zero-valued for positional arguments.
	Name string

	// Value is the string value of an argument. It has been validated to match the Type.
	// See the ArgType const godoc for further details on how to parse the value for the
	// Type.
	Value string

	// Type identifies the type of the argument.
	Type ArgType
}

func (a Arg) String() string {
	buf := strings.Builder{}
	if len(a.Name) > 0 {
		buf.WriteString(a.Name)
		buf.WriteString(": ")
	}
	if a.Type == ArgTypeString {
		buf.WriteString(strconv.Quote(a.Value))
	} else {
		buf.WriteString(a.Value)
	}
	return buf.String()
}

// ArgType is an argument's type.
type ArgType string

const (
	// ArgTypeString identifies string values.
	ArgTypeString ArgType = "string"

	// ArgTypeInt identifies int values. Values of this type may be in decimal,
	// octal, hex or binary string representations. Consider using strconv.ParseInt
	// to parse, as it supports all these string representations.
	ArgTypeInt ArgType = "int"

	// ArgTypeBool identifies bool values. Values of this type must either be the
	// string "true" or "false".
	ArgTypeBool ArgType = "bool"
)

// ValueType is a tag's value type.
type ValueType string

const (
	// ValueTypeNone indicates that the tag has no value.
	ValueTypeNone ValueType = ""

	// ValueTypeString identifies string values.
	ValueTypeString ValueType = "string"

	// ValueTypeInt identifies int values. Values of this type may be in decimal,
	// octal, hex or binary string representations. Consider using strconv.ParseInt
	// to parse, as it supports all these string representations.
	ValueTypeInt ValueType = "int"

	// ValueTypeBool identifies bool values. Values of this type must either be the
	// string "true" or "false".
	ValueTypeBool ValueType = "bool"

	// ValueTypeTag identifies that the value is another tag.
	ValueTypeTag ValueType = "tag"

	// ValueTypeRaw identifies that the value is raw, untyped content and contains
	// all text from the tag declaration following the "=" sign, up to the last
	// non-whitespace character.
	ValueTypeRaw ValueType = "raw"
)
