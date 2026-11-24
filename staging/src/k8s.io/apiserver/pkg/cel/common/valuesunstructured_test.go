/*
Copyright The Kubernetes Authors.

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

package common_test

import (
	"reflect"
	"sort"
	"testing"

	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/traits"

	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// countingSchema records calls to Properties() so tests can assert that
// single-field lookups go through Property()/HasProperties() instead of
// materializing the full properties map.
type countingSchema struct {
	common.Schema
	propertiesCalls int
}

func (s *countingSchema) Properties() map[string]common.Schema {
	s.propertiesCalls++
	return s.Schema.Properties()
}

func objSchema(props map[string]spec.Schema) *spec.Schema {
	return &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, Properties: props}}
}

func mapKeys(t *testing.T, m traits.Mapper) []string {
	t.Helper()
	var keys []string
	for it := m.Iterator(); it.HasNext() == types.True; {
		keys = append(keys, string(it.Next().(types.String)))
	}
	sort.Strings(keys)
	return keys
}

func TestUnstructuredToValPropertyLookup(t *testing.T) {
	str := *spec.StringProperty()
	nullableStr := *spec.StringProperty()
	nullableStr.Nullable = true

	embedded := objSchema(nil)
	embedded.Extensions = spec.Extensions{"x-kubernetes-embedded-resource": true}
	embeddedWithProps := objSchema(map[string]spec.Schema{"spec": *objSchema(nil)})
	embeddedWithProps.Extensions = spec.Extensions{"x-kubernetes-embedded-resource": true}

	cases := []struct {
		name     string
		schema   *spec.Schema
		value    map[string]interface{}
		found    map[string]interface{} // escaped key -> expected native value
		notFound []string
		// Expected iterator keys; nil skips the check. Only set for values without
		// undeclared fields: unstructuredMap.Iterator() currently yields a trailing
		// nil key for each field that has no schema.
		keys []string
	}{
		{
			name:     "nil properties without additionalProperties exposes no fields",
			schema:   objSchema(nil),
			value:    map[string]interface{}{"a": "x"},
			notFound: []string{"a"},
		},
		{
			name:     "empty non-nil properties exposes no fields",
			schema:   objSchema(map[string]spec.Schema{}),
			value:    map[string]interface{}{"a": "x"},
			notFound: []string{"a"},
		},
		{
			name:   "declared properties are found by escaped name",
			schema: objSchema(map[string]spec.Schema{"a-b": str, "if": str, "plain": str}),
			value:  map[string]interface{}{"a-b": "x", "if": "y", "plain": "z", "undeclared": "u"},
			found: map[string]interface{}{
				"a__dash__b": "x",
				"__if__":     "y",
				"plain":      "z",
			},
			notFound: []string{"undeclared", "__bogus__"},
		},
		{
			name:   "iterator yields escaped names of declared properties",
			schema: objSchema(map[string]spec.Schema{"a-b": str, "if": str, "plain": str}),
			value:  map[string]interface{}{"a-b": "x", "if": "y", "plain": "z"},
			keys:   []string{"__if__", "a__dash__b", "plain"},
		},
		{
			name:     "declared property absent from value is not found",
			schema:   objSchema(map[string]spec.Schema{"a": str, "b": str}),
			value:    map[string]interface{}{"a": "x"},
			found:    map[string]interface{}{"a": "x"},
			notFound: []string{"b"},
			keys:     []string{"a"},
		},
		{
			name:     "null nullable property is treated as absent",
			schema:   objSchema(map[string]spec.Schema{"n": nullableStr}),
			value:    map[string]interface{}{"n": nil},
			notFound: []string{"n"},
		},
		{
			name: "additionalProperties keys are not unescaped",
			schema: &spec.Schema{SchemaProps: spec.SchemaProps{
				Type:                 []string{"object"},
				AdditionalProperties: &spec.SchemaOrBool{Allows: true, Schema: &str},
			}},
			value:    map[string]interface{}{"a-b": "x"},
			found:    map[string]interface{}{"a-b": "x"},
			notFound: []string{"a__dash__b"},
			keys:     []string{"a-b"},
		},
		{
			name:   "embedded resource without properties exposes type and object meta",
			schema: embedded,
			value: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Thing",
				"metadata":   map[string]interface{}{"name": "n"},
				"spec":       map[string]interface{}{},
			},
			found: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Thing",
				"metadata":   map[string]interface{}{"name": "n"},
			},
			notFound: []string{"spec"},
		},
		{
			name:   "embedded resource with properties keeps declared and adds type and object meta",
			schema: embeddedWithProps,
			value: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Thing",
				"spec":       map[string]interface{}{},
			},
			found: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Thing",
				"spec":       map[string]interface{}{},
			},
			notFound: []string{"metadata"},
			keys:     []string{"apiVersion", "kind", "spec"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := common.UnstructuredToVal(tc.value, &openapi.Schema{Schema: tc.schema})
			m, ok := v.(traits.Mapper)
			if !ok {
				t.Fatalf("UnstructuredToVal() = %v (%T), want traits.Mapper", v, v)
			}
			for k, want := range tc.found {
				got, found := m.Find(types.String(k))
				if !found {
					t.Errorf("Find(%q) not found, want %v", k, want)
					continue
				}
				if !reflect.DeepEqual(got.Value(), want) {
					t.Errorf("Find(%q) = %v, want %v", k, got.Value(), want)
				}
				if c := m.Contains(types.String(k)); c != types.True {
					t.Errorf("Contains(%q) = %v, want true", k, c)
				}
			}
			for _, k := range tc.notFound {
				if got, found := m.Find(types.String(k)); found {
					t.Errorf("Find(%q) = %v, want not found", k, got)
				}
				if c := m.Contains(types.String(k)); c != types.False {
					t.Errorf("Contains(%q) = %v, want false", k, c)
				}
			}
			if tc.keys != nil {
				got := mapKeys(t, m)
				if !reflect.DeepEqual(got, tc.keys) {
					t.Errorf("Iterator() keys = %v, want %v", got, tc.keys)
				}
			}
		})
	}
}

// TestUnstructuredToValAvoidsProperties guards against regressing to
// Properties() on hot paths, which copies the whole properties map per call.
func TestUnstructuredToValAvoidsProperties(t *testing.T) {
	str := *spec.StringProperty()
	s := &countingSchema{Schema: &openapi.Schema{Schema: objSchema(map[string]spec.Schema{"name": str, "if": str})}}

	m := common.UnstructuredToVal(map[string]interface{}{"name": "a", "if": "b"}, s).(traits.Mapper)
	m.Find(types.String("name"))
	m.Find(types.String("__if__"))
	m.Contains(types.String("missing"))
	m.Get(types.String("name"))
	mapKeys(t, m)
	if s.propertiesCalls != 0 {
		t.Errorf("Properties() called %d times, want 0", s.propertiesCalls)
	}
}
