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

package defaulting

import (
	"k8s.io/apiserver/pkg/cel/common"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// isNonNullalbeNull returns true if the item is nil AND it's nullable
func isNonNullableNull(x value.Value, s common.Schema) bool {
	return (x == nil || x.IsNull()) && s != nil && s.Nullable() == false
}

// Default does defaulting of x depending on default values in s.
// Default values from s are deep-copied.
//
// PruneNonNullableNullsWithoutDefaults has left the non-nullable nulls
// that have a default here.
func Default(x value.Value, s common.Schema) {
	if s == nil {
		return
	}

	switch {
	case x.IsMap():
		asMap := x.AsMap()
		properties := s.Properties()
		for k, prop := range properties {
			def := prop.Default()
			if def == nil {
				continue
			}

			if v, found := asMap.Get(k); !found || isNonNullableNull(v, prop) {
				asMap.Set(k, value.NewValueInterface(def))
			}
		}

		asMap.Iterate(func(k string, v value.Value) bool {
			if prop, found := properties[k]; found {
				Default(v, prop)
			} else if add := s.AdditionalProperties(); add != nil {
				if isNonNullableNull(v, add.Schema()) {
					asMap.Set(k, value.NewValueInterface(add.Schema().Default()))
				}

				Default(v, add.Schema())
			}
			return true
		})
	case x.IsList():
		asList := x.AsList()
		iter := asList.Range()

		for iter.Next() {
			i, v := iter.Item()
			if isNonNullableNull(v, s.Items()) {
				asList.Set(i, value.NewValueInterface(s.Items().Default()))
			}
			Default(v, s.Items())
		}
	default:
		// scalars, do nothing
	}
}
