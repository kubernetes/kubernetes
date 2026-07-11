/*
Copyright 2022 The Kubernetes Authors.

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

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

func TestMapList(t *testing.T) {
	for _, tc := range []struct {
		name          string
		sts           schema.Structural
		items         []interface{}
		warmUpQueries []interface{}
		query         interface{}
		expected      interface{}
	}{
		{
			name: "default list type",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
			},
			query:    map[string]interface{}{},
			expected: nil,
		},
		{
			name: "non list type",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "map",
				},
			},
			query:    map[string]interface{}{},
			expected: nil,
		},
		{
			name: "non-map list type",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Extensions: schema.Extensions{
					XListType: &listTypeSet,
				},
			},
			query:    map[string]interface{}{},
			expected: nil,
		},
		{
			name: "no keys",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Extensions: schema.Extensions{
					XListType: &listTypeMap,
				},
			},
			query:    map[string]interface{}{},
			expected: nil,
		},
		{
			name: "single key",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Extensions: schema.Extensions{
					XListType:    &listTypeMap,
					XListMapKeys: []string{"k"},
				},
			},
			items: []interface{}{
				map[string]interface{}{
					"k":  "a",
					"v1": "a",
				},
				map[string]interface{}{
					"k":  "b",
					"v1": "b",
				},
			},
			query: map[string]interface{}{
				"k":  "b",
				"v1": "B",
			},
			expected: map[string]interface{}{
				"k":  "b",
				"v1": "b",
			},
		},
		{
			name: "single key ignoring non-map query",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Extensions: schema.Extensions{
					XListType:    &listTypeMap,
					XListMapKeys: []string{"k"},
				},
			},
			items: []interface{}{
				map[string]interface{}{
					"k":  "a",
					"v1": "a",
				},
			},
			query:    42,
			expected: nil,
		},
		{
			name: "single key ignoring unkeyable query",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Extensions: schema.Extensions{
					XListType:    &listTypeMap,
					XListMapKeys: []string{"k"},
				},
			},
			items: []interface{}{
				map[string]interface{}{
					"k":  "a",
					"v1": "a",
				},
			},
			query: map[string]interface{}{
				"k": map[string]interface{}{
					"keys": "must",
					"be":   "scalars",
				},
				"v1": "A",
			},
			expected: nil,
		},
		{
			name: "ignores item of invalid type",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Extensions: schema.Extensions{
					XListType:    &listTypeMap,
					XListMapKeys: []string{"k"},
				},
			},
			items: []interface{}{
				map[string]interface{}{
					"k":  "a",
					"v1": "a",
				},
				5,
			},
			query: map[string]interface{}{
				"k":  "a",
				"v1": "A",
			},
			expected: map[string]interface{}{
				"k":  "a",
				"v1": "a",
			},
		},
		{
			name: "keep first entry when duplicated keys are encountered",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Extensions: schema.Extensions{
					XListType:    &listTypeMap,
					XListMapKeys: []string{"k"},
				},
			},
			items: []interface{}{
				map[string]interface{}{
					"k":  "a",
					"v1": "a",
				},
				map[string]interface{}{
					"k":  "a",
					"v1": "b",
				},
			},
			query: map[string]interface{}{
				"k":  "a",
				"v1": "A",
			},
			expected: map[string]interface{}{
				"k":  "a",
				"v1": "a",
			},
		},
		{
			name: "keep first entry when duplicated multi-keys are encountered",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Extensions: schema.Extensions{
					XListType:    &listTypeMap,
					XListMapKeys: []string{"k1", "k2"},
				},
			},
			items: []interface{}{
				map[string]interface{}{
					"k1": "a",
					"k2": "b",
					"v1": "a",
				},
				map[string]interface{}{
					"k1": "a",
					"k2": "b",
					"v1": "b",
				},
				map[string]interface{}{
					"k1": "x",
					"k2": "y",
					"v1": "z",
				},
			},
			warmUpQueries: []interface{}{
				map[string]interface{}{
					"k1": "x",
					"k2": "y",
				},
			},
			query: map[string]interface{}{
				"k1": "a",
				"k2": "b",
			},
			expected: map[string]interface{}{
				"k1": "a",
				"k2": "b",
				"v1": "a",
			},
		},
		{
			name: "multiple keys with defaults ignores item with nil value for key",
			sts: schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Extensions: schema.Extensions{
					XListType:    &listTypeMap,
					XListMapKeys: []string{"kb", "kf", "ki", "ks"},
				},
				Properties: map[string]schema.Structural{
					"kb": {
						Generic: schema.Generic{
							Default: schema.JSON{Object: true},
						},
					},
					"kf": {
						Generic: schema.Generic{
							Default: schema.JSON{Object: float64(2.0)},
						},
					},
					"ki": {
						Generic: schema.Generic{
							Default: schema.JSON{Object: int64(42)},
						},
					},
					"ks": {
						Generic: schema.Generic{
							Default: schema.JSON{Object: "hello"},
						},
					},
				},
			},
			items: []interface{}{
				map[string]interface{}{
					"kb": nil,
					"kf": float64(2.0),
					"ki": int64(42),
					"ks": "hello",
					"v1": "a",
				},
				map[string]interface{}{
					"kb": false,
					"kf": float64(2.0),
					"ki": int64(42),
					"ks": "hello",
					"v1": "b",
				},
			},
			query: map[string]interface{}{
				"kb": false,
				"kf": float64(2.0),
				"ki": int64(42),
				"ks": "hello",
				"v1": "B",
			},
			expected: map[string]interface{}{
				"kb": false,
				"kf": float64(2.0),
				"ki": int64(42),
				"ks": "hello",
				"v1": "b",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mapList := makeMapList(&tc.sts, tc.items)
			for _, warmUp := range tc.warmUpQueries {
				mapList.Get(warmUp)
			}
			actual := mapList.Get(tc.query)
			if !reflect.DeepEqual(tc.expected, actual) {
				t.Errorf("got: %v, expected %v", actual, tc.expected)
			}
		})
	}
}
