/*
Copyright 2021 The Kubernetes Authors.

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

package cel

import (
	"reflect"
	"testing"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

var (
	listTypeSet  = "set"
	listTypeMap  = "map"
	stringSchema = schema.Structural{
		Generic: schema.Generic{
			Type: "string",
		},
	}
	intSchema = schema.Structural{
		Generic: schema.Generic{
			Type: "integer",
		},
		ValueValidation: &schema.ValueValidation{
			Format: "int64",
		},
	}
	mapListElementSchema = schema.Structural{
		Generic: schema.Generic{
			Type: "object",
		},
		Properties: map[string]schema.Structural{
			"key": stringSchema,
			"val": intSchema,
		},
	}
	mapListSchema = schema.Structural{
		Extensions: schema.Extensions{XListType: &listTypeMap, XListMapKeys: []string{"key"}},
		Generic: schema.Generic{
			Type: "array",
		},
		Items: &mapListElementSchema,
	}
	multiKeyMapListSchema = schema.Structural{
		Extensions: schema.Extensions{XListType: &listTypeMap, XListMapKeys: []string{"key1", "key2"}},
		Generic: schema.Generic{
			Type: "array",
		},
		Items: &schema.Structural{
			Generic: schema.Generic{
				Type: "object",
			},
			Properties: map[string]schema.Structural{
				"key1": stringSchema,
				"key2": stringSchema,
				"val":  intSchema,
			},
		},
	}
	setListSchema = schema.Structural{
		Extensions: schema.Extensions{XListType: &listTypeSet},
		Generic: schema.Generic{
			Type: "array",
		},
		Items: &stringSchema,
	}
	atomicListSchema = schema.Structural{
		Generic: schema.Generic{
			Type: "array",
		},
		Items: &stringSchema,
	}
	objectSchema = schema.Structural{
		Generic: schema.Generic{
			Type: "object",
		},
		Properties: map[string]schema.Structural{
			"field1": stringSchema,
			"field2": stringSchema,
		},
	}
	mapSchema = schema.Structural{
		Generic: schema.Generic{
			Type: "object",
		},
		AdditionalProperties: &schema.StructuralOrBool{
			Bool:       true,
			Structural: &stringSchema,
		},
	}
)

func TestEquality(t *testing.T) {
	cases := []struct {
		name  string
		lhs   ref.Val
		rhs   ref.Val
		equal bool
	}{
		{
			name: "map lists are equal regardless of order",
			lhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key": "a",
					"val": 1,
				},
				map[string]interface{}{
					"key": "b",
					"val": 2,
				},
			}, &mapListSchema),
			rhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key": "b",
					"val": 2,
				},
				map[string]interface{}{
					"key": "a",
					"val": 1,
				},
			}, &mapListSchema),
			equal: true,
		},
		{
			name: "map lists are not equal if contents differs",
			lhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key": "a",
					"val": 1,
				},
				map[string]interface{}{
					"key": "b",
					"val": 2,
				},
			}, &mapListSchema),
			rhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key": "a",
					"val": 1,
				},
				map[string]interface{}{
					"key": "b",
					"val": 3,
				},
			}, &mapListSchema),
			equal: false,
		},
		{
			name: "map lists are not equal if length differs",
			lhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key": "a",
					"val": 1,
				},
				map[string]interface{}{
					"key": "b",
					"val": 2,
				},
			}, &mapListSchema),
			rhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key": "a",
					"val": 1,
				},
				map[string]interface{}{
					"key": "b",
					"val": 2,
				},
				map[string]interface{}{
					"key": "c",
					"val": 3,
				},
			}, &mapListSchema),
			equal: false,
		},
		{
			name: "multi-key map lists are equal regardless of order",
			lhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key1": "a1",
					"key2": "a2",
					"val":  1,
				},
				map[string]interface{}{
					"key1": "b1",
					"key2": "b2",
					"val":  2,
				},
			}, &multiKeyMapListSchema),
			rhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key1": "b1",
					"key2": "b2",
					"val":  2,
				},
				map[string]interface{}{
					"key1": "a1",
					"key2": "a2",
					"val":  1,
				},
			}, &multiKeyMapListSchema),
			equal: true,
		},
		{
			name: "multi-key map lists with different contents are not equal",
			lhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key1": "a1",
					"key2": "a2",
					"val":  1,
				},
				map[string]interface{}{
					"key1": "b1",
					"key2": "b2",
					"val":  2,
				},
			}, &multiKeyMapListSchema),
			rhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key1": "a1",
					"key2": "a2",
					"val":  1,
				},
				map[string]interface{}{
					"key1": "b1",
					"key2": "b2",
					"val":  3,
				},
			}, &multiKeyMapListSchema),
			equal: false,
		},
		{
			name: "multi-key map lists with different keys are not equal",
			lhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key1": "a1",
					"key2": "a2",
					"val":  1,
				},
				map[string]interface{}{
					"key1": "b1",
					"key2": "b2",
					"val":  2,
				},
			}, &multiKeyMapListSchema),
			rhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key1": "a1",
					"key2": "a2",
					"val":  1,
				},
				map[string]interface{}{
					"key1": "c1",
					"key2": "c2",
					"val":  3,
				},
			}, &multiKeyMapListSchema),
			equal: false,
		},
		{
			name: "multi-key map lists with different lengths are not equal",
			lhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key1": "a1",
					"key2": "a2",
					"val":  1,
				},
			}, &multiKeyMapListSchema),
			rhs: UnstructuredToVal([]interface{}{
				map[string]interface{}{
					"key1": "a1",
					"key2": "a2",
					"val":  1,
				},
				map[string]interface{}{
					"key1": "b1",
					"key2": "b2",
					"val":  3,
				},
			}, &multiKeyMapListSchema),
			equal: false,
		},
		{
			name:  "set lists are equal regardless of order",
			lhs:   UnstructuredToVal([]interface{}{"a", "b"}, &setListSchema),
			rhs:   UnstructuredToVal([]interface{}{"b", "a"}, &setListSchema),
			equal: true,
		},
		{
			name:  "set lists are not equal if contents differ",
			lhs:   UnstructuredToVal([]interface{}{"a", "b"}, &setListSchema),
			rhs:   UnstructuredToVal([]interface{}{"a", "c"}, &setListSchema),
			equal: false,
		},
		{
			name:  "set lists are not equal if lengths differ",
			lhs:   UnstructuredToVal([]interface{}{"a", "b"}, &setListSchema),
			rhs:   UnstructuredToVal([]interface{}{"a", "b", "c"}, &setListSchema),
			equal: false,
		},
		{
			name:  "identical atomic lists are equal",
			lhs:   UnstructuredToVal([]interface{}{"a", "b"}, &atomicListSchema),
			rhs:   UnstructuredToVal([]interface{}{"a", "b"}, &atomicListSchema),
			equal: true,
		},
		{
			name:  "atomic lists are not equal if order differs",
			lhs:   UnstructuredToVal([]interface{}{"a", "b"}, &atomicListSchema),
			rhs:   UnstructuredToVal([]interface{}{"b", "a"}, &atomicListSchema),
			equal: false,
		},
		{
			name:  "atomic lists are not equal if contents differ",
			lhs:   UnstructuredToVal([]interface{}{"a", "b"}, &atomicListSchema),
			rhs:   UnstructuredToVal([]interface{}{"a", "c"}, &atomicListSchema),
			equal: false,
		},
		{
			name:  "atomic lists are not equal if lengths differ",
			lhs:   UnstructuredToVal([]interface{}{"a", "b"}, &atomicListSchema),
			rhs:   UnstructuredToVal([]interface{}{"a", "b", "c"}, &atomicListSchema),
			equal: false,
		},
		{
			name:  "identical objects are equal",
			lhs:   UnstructuredToVal(map[string]interface{}{"field1": "a", "field2": "b"}, &objectSchema),
			rhs:   UnstructuredToVal(map[string]interface{}{"field1": "a", "field2": "b"}, &objectSchema),
			equal: true,
		},
		{
			name:  "objects are equal regardless of field order",
			lhs:   UnstructuredToVal(map[string]interface{}{"field1": "a", "field2": "b"}, &objectSchema),
			rhs:   UnstructuredToVal(map[string]interface{}{"field2": "b", "field1": "a"}, &objectSchema),
			equal: true,
		},
		{
			name:  "objects are not equal if contents differs",
			lhs:   UnstructuredToVal(map[string]interface{}{"field1": "a", "field2": "b"}, &objectSchema),
			rhs:   UnstructuredToVal(map[string]interface{}{"field1": "a", "field2": "c"}, &objectSchema),
			equal: false,
		},
		{
			name:  "objects are not equal if length differs",
			lhs:   UnstructuredToVal(map[string]interface{}{"field1": "a", "field2": "b"}, &objectSchema),
			rhs:   UnstructuredToVal(map[string]interface{}{"field1": "a"}, &objectSchema),
			equal: false,
		},
		{
			name:  "identical maps are equal",
			lhs:   UnstructuredToVal(map[string]interface{}{"key1": "a", "key2": "b"}, &mapSchema),
			rhs:   UnstructuredToVal(map[string]interface{}{"key1": "a", "key2": "b"}, &mapSchema),
			equal: true,
		},
		{
			name:  "maps are equal regardless of field order",
			lhs:   UnstructuredToVal(map[string]interface{}{"key1": "a", "key2": "b"}, &mapSchema),
			rhs:   UnstructuredToVal(map[string]interface{}{"key2": "b", "key1": "a"}, &mapSchema),
			equal: true,
		},
		{
			name:  "maps are not equal if contents differs",
			lhs:   UnstructuredToVal(map[string]interface{}{"key1": "a", "key2": "b"}, &mapSchema),
			rhs:   UnstructuredToVal(map[string]interface{}{"key1": "a", "key2": "c"}, &mapSchema),
			equal: false,
		},
		{
			name:  "maps are not equal if length differs",
			lhs:   UnstructuredToVal(map[string]interface{}{"key1": "a", "key2": "b"}, &mapSchema),
			rhs:   UnstructuredToVal(map[string]interface{}{"key1": "a", "key2": "b", "key3": "c"}, &mapSchema),
			equal: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Compare types with schema against themselves
			if tc.lhs.Equal(tc.rhs) != types.Bool(tc.equal) {
				t.Errorf("expected Equals to return %v", tc.equal)
			}
			if tc.rhs.Equal(tc.lhs) != types.Bool(tc.equal) {
				t.Errorf("expected Equals to return %v", tc.equal)
			}

			// Compare types with schema against native types. This is slightly different than how
			// CEL performs equality against data literals, but is a good sanity check.
			if tc.lhs.Equal(types.DefaultTypeAdapter.NativeToValue(tc.rhs.Value())) != types.Bool(tc.equal) {
				t.Errorf("expected unstructuredVal.Equals(<native type>) to return %v", tc.equal)
			}
			if tc.rhs.Equal(types.DefaultTypeAdapter.NativeToValue(tc.lhs.Value())) != types.Bool(tc.equal) {
				t.Errorf("expected unstructuredVal.Equals(<native type>) to return %v", tc.equal)
			}
		})
	}
}

func TestLister(t *testing.T) {
	cases := []struct {
		name         string
		unstructured []interface{}
		schema       *schema.Structural
		itemSchema   *schema.Structural
		size         int64
		notContains  []ref.Val
		addition     []interface{}
		expectAdded  []interface{}
	}{
		{
			name: "map list",
			unstructured: []interface{}{
				map[string]interface{}{
					"key": "a",
					"val": 1,
				},
				map[string]interface{}{
					"key": "b",
					"val": 2,
				},
			},
			schema:     &mapListSchema,
			itemSchema: &mapListElementSchema,
			size:       2,
			notContains: []ref.Val{
				UnstructuredToVal(map[string]interface{}{
					"key": "a",
					"val": 2,
				}, &mapListElementSchema),
				UnstructuredToVal(map[string]interface{}{
					"key": "c",
					"val": 1,
				}, &mapListElementSchema),
			},
			addition: []interface{}{
				map[string]interface{}{
					"key": "b",
					"val": 3,
				},
				map[string]interface{}{
					"key": "c",
					"val": 4,
				},
			},
			expectAdded: []interface{}{
				map[string]interface{}{
					"key": "a",
					"val": 1,
				},
				map[string]interface{}{
					"key": "b",
					"val": 3,
				},
				map[string]interface{}{
					"key": "c",
					"val": 4,
				},
			},
		},
		{
			name:         "set list",
			unstructured: []interface{}{"a", "b"},
			schema:       &setListSchema,
			itemSchema:   &stringSchema,
			size:         2,
			notContains:  []ref.Val{UnstructuredToVal("c", &stringSchema)},
			addition:     []interface{}{"b", "c"},
			expectAdded:  []interface{}{"a", "b", "c"},
		},
		{
			name:         "atomic list",
			unstructured: []interface{}{"a", "b"},
			schema:       &atomicListSchema,
			itemSchema:   &stringSchema,
			size:         2,
			notContains:  []ref.Val{UnstructuredToVal("c", &stringSchema)},
			addition:     []interface{}{"b", "c"},
			expectAdded:  []interface{}{"a", "b", "b", "c"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			lister := UnstructuredToVal(tc.unstructured, tc.schema).(traits.Lister)
			if lister.Size().Value() != tc.size {
				t.Errorf("Expected Size to return %d but got %d", tc.size, lister.Size().Value())
			}
			iter := lister.Iterator()
			for i := 0; i < int(tc.size); i++ {
				get := lister.Get(types.Int(i)).Value()
				if !reflect.DeepEqual(get, tc.unstructured[i]) {
					t.Errorf("Expected Get to return %v for index %d but got %v", tc.unstructured[i], i, get)
				}
				if iter.HasNext() != types.True {
					t.Error("Expected HasNext to return true")
				}
				next := iter.Next().Value()
				if !reflect.DeepEqual(next, tc.unstructured[i]) {
					t.Errorf("Expected Next to return %v for index %d but got %v", tc.unstructured[i], i, next)
				}
			}
			if iter.HasNext() != types.False {
				t.Error("Expected HasNext to return false")
			}
			for _, contains := range tc.unstructured {
				if lister.Contains(UnstructuredToVal(contains, tc.itemSchema)) != types.True {
					t.Errorf("Expected Contains to return true for %v", contains)
				}
			}
			for _, notContains := range tc.notContains {
				if lister.Contains(notContains) != types.False {
					t.Errorf("Expected Contains to return false for %v", notContains)
				}
			}

			addition := UnstructuredToVal(tc.addition, tc.schema).(traits.Lister)
			added := lister.Add(addition).Value()
			if !reflect.DeepEqual(added, tc.expectAdded) {
				t.Errorf("Expected Add to return %v but got %v", tc.expectAdded, added)
			}
		})
	}
}

func TestMapper(t *testing.T) {
	cases := []struct {
		name           string
		unstructured   map[string]interface{}
		schema         *schema.Structural
		propertySchema func(key string) (*schema.Structural, bool)
		size           int64
		notContains    []ref.Val
	}{
		{
			name: "object",
			unstructured: map[string]interface{}{
				"field1": "a",
				"field2": "b",
			},
			schema: &objectSchema,
			propertySchema: func(key string) (*schema.Structural, bool) {
				if s, ok := objectSchema.Properties[key]; ok {
					return &s, true
				}
				return nil, false
			},
			size: 2,
			notContains: []ref.Val{
				UnstructuredToVal("field3", &stringSchema),
			},
		},
		{
			name: "map",
			unstructured: map[string]interface{}{
				"key1": "a",
				"key2": "b",
			},
			schema:         &mapSchema,
			propertySchema: func(key string) (*schema.Structural, bool) { return mapSchema.AdditionalProperties.Structural, true },
			size:           2,
			notContains: []ref.Val{
				UnstructuredToVal("key3", &stringSchema),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mapper := UnstructuredToVal(tc.unstructured, tc.schema).(traits.Mapper)
			if mapper.Size().Value() != tc.size {
				t.Errorf("Expected Size to return %d but got %d", tc.size, mapper.Size().Value())
			}
			iter := mapper.Iterator()
			iterResults := map[interface{}]struct{}{}
			keys := map[interface{}]struct{}{}
			for k := range tc.unstructured {
				keys[k] = struct{}{}
				get := mapper.Get(types.String(k)).Value()
				if !reflect.DeepEqual(get, tc.unstructured[k]) {
					t.Errorf("Expected Get to return %v for key %s but got %v", tc.unstructured[k], k, get)
				}
				if iter.HasNext() != types.True {
					t.Error("Expected HasNext to return true")
				}
				iterResults[iter.Next().Value()] = struct{}{}
			}
			if !reflect.DeepEqual(iterResults, keys) {
				t.Errorf("Expected accumulation of iterator.Next calls to be %v but got %v", keys, iterResults)
			}
			if iter.HasNext() != types.False {
				t.Error("Expected HasNext to return false")
			}
			for contains := range tc.unstructured {
				if mapper.Contains(UnstructuredToVal(contains, &stringSchema)) != types.True {
					t.Errorf("Expected Contains to return true for %v", contains)
				}
			}
			for _, notContains := range tc.notContains {
				if mapper.Contains(notContains) != types.False {
					t.Errorf("Expected Contains to return false for %v", notContains)
				}
			}
		})
	}
}

func BenchmarkUnstructuredToVal(b *testing.B) {
	u := []interface{}{
		map[string]interface{}{
			"key": "a",
			"val": 1,
		},
		map[string]interface{}{
			"key": "b",
			"val": 2,
		},
		map[string]interface{}{
			"key": "@b",
			"val": 2,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		if val := UnstructuredToVal(u, &mapListSchema); val == nil {
			b.Fatal(val)
		}
	}
}

func BenchmarkUnstructuredToValWithEscape(b *testing.B) {
	u := []interface{}{
		map[string]interface{}{
			"key": "a.1",
			"val": "__i.1",
		},
		map[string]interface{}{
			"key": "b.1",
			"val": 2,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		if val := UnstructuredToVal(u, &mapListSchema); val == nil {
			b.Fatal(val)
		}
	}
}
