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
	"fmt"
	"strings"
)

// TypedTag represents a single comment tag with typed args.
type TypedTag struct {
	// Name is the name of the tag with no arguments.
	Name string
	// Args is a list of optional arguments to the tag.
	Args []Arg
	// Value is the value of the tag.
	Value string
}

func (t TypedTag) String() string {
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
	if len(t.Value) > 0 {
		buf.WriteString("=")
		buf.WriteString(t.Value)
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

// String returns the string representation of the argument.
func (a Arg) String() string {
	if len(a.Name) > 0 {
		return fmt.Sprintf("%s: %v", a.Name, a.Value)
	}
	return fmt.Sprintf("%v", a.Value)
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
