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

import structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"

func isNonNullableNonDefaultableNull(x interface{}, s *structuralschema.Structural) bool {
	return x == nil && s != nil && s.Generic.Nullable == false && s.Default.Object == nil
}

func getSchemaForField(field string, s *structuralschema.Structural) *structuralschema.Structural {
	if s == nil {
		return nil
	}
	schema, ok := s.Properties[field]
	if ok {
		return &schema
	}
	if s.AdditionalProperties != nil {
		return s.AdditionalProperties.Structural
	}
	return nil
}

// PruneNonNullableNullsWithoutDefaults removes non-nullable
// non-defaultable null values from object.
//
// Non-nullable nulls that have a default are left alone here and will
// be defaulted later.
func PruneNonNullableNullsWithoutDefaults(x interface{}, s *structuralschema.Structural) {
	switch x := x.(type) {
	case map[string]interface{}:
		for k, v := range x {
			schema := getSchemaForField(k, s)
			if isNonNullableNonDefaultableNull(v, schema) {
				delete(x, k)
			} else {
				PruneNonNullableNullsWithoutDefaults(v, schema)
			}
		}
	case []interface{}:
		var schema *structuralschema.Structural
		if s != nil {
			schema = s.Items
		}
		for i := range x {
			PruneNonNullableNullsWithoutDefaults(x[i], schema)
		}
	default:
		// scalars, do nothing
	}
}
