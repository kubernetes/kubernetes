/*
Copyright 2020 The Kubernetes Authors.

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

func isNonNullableNonDefaultableNull(x interface{}, s common.Schema) bool {
	return x == nil && s != nil && s.Nullable() == false && s.Default() == nil
}

func getSchemaForField(field string, s common.Schema) common.Schema {
	if s == nil {
		return nil
	}
	schema, ok := s.Properties()[field]
	if ok {
		return schema
	}
	if add := s.AdditionalProperties(); add != nil {
		if addSchema := add.Schema(); add != nil {
			return addSchema
		} else if add.Allows() {
			panic("TODO: return wildcard schema")
		}
	}
	return nil
}

// PruneNonNullableNullsWithoutDefaults removes non-nullable
// non-defaultable null values from object.
//
// Non-nullable nulls that have a default are left alone here and will
// be defaulted later.
func PruneNonNullableNullsWithoutDefaults(x value.Value, s common.Schema) {
	switch {
	case x.IsMap():
		asMap := x.AsMap()
		asMap.Iterate(func(k string, v value.Value) bool {
			schema := getSchemaForField(k, s)
			if isNonNullableNonDefaultableNull(v, schema) {
				asMap.Delete(k)
			} else {
				PruneNonNullableNullsWithoutDefaults(v, schema)
			}
			return true
		})
	case x.IsList():
		var schema common.Schema
		if s != nil {
			schema = s.Items()
		}

		asList := x.AsList()
		iter := asList.Range()

		for iter.Next() {
			_, v := iter.Item()
			PruneNonNullableNullsWithoutDefaults(v, schema)
		}
	default:
		// scalars, do nothing
	}
}
