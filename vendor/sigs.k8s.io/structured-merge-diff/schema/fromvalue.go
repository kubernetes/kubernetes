/*
Copyright 2019 The Kubernetes Authors.

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

package schema

import (
	"sigs.k8s.io/structured-merge-diff/value"
)

// TypeRefFromValue creates an inlined type from a value v
func TypeRefFromValue(v value.Value) TypeRef {
	atom := atomFor(v)
	return TypeRef{
		Inlined: atom,
	}
}

func atomFor(v value.Value) Atom {
	switch {
	// Untyped cases (handled at the bottom of this function)
	case v.Null:
	case v.ListValue != nil:
	case v.FloatValue != nil:
	case v.IntValue != nil:
	case v.StringValue != nil:
	case v.BooleanValue != nil:
	// Recursive case
	case v.MapValue != nil:
		s := Struct{}
		for i := range v.MapValue.Items {
			child := v.MapValue.Items[i]
			field := StructField{
				Name: child.Name,
				Type: TypeRef{
					Inlined: atomFor(child.Value),
				},
			}
			s.Fields = append(s.Fields, field)
		}
		return Atom{Struct: &s}
	}

	return Atom{Untyped: &Untyped{}}
}
