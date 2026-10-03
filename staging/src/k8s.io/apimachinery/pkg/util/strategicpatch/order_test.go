/*
Copyright 2026 The Kubernetes Authors.

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

package strategicpatch

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

func TestSliceOrderIndex(t *testing.T) {
	testCases := []struct {
		name  string
		order []interface{}
		item  interface{}
		kind  reflect.Kind
		want  int
	}{
		{name: "first duplicate", order: []interface{}{"a", "b", "a"}, item: "a", kind: reflect.String, want: 0},
		{name: "absent", order: []interface{}{"a"}, item: "missing", kind: reflect.String, want: -1},
		{name: "nil", order: []interface{}{nil, "a"}, kind: reflect.String, want: 0},
		{name: "numeric types", order: []interface{}{int64(1), float64(1)}, item: float64(1), kind: reflect.Float64, want: 1},
		{name: "NaN", order: []interface{}{math.NaN()}, item: math.NaN(), kind: reflect.Float64, want: -1},
		{name: "signed zero", order: []interface{}{float64(0), math.Copysign(0, -1)}, item: math.Copysign(0, -1), kind: reflect.Float64, want: 0},
		{name: "map duplicate", order: []interface{}{map[string]interface{}{"name": "a"}, map[string]interface{}{"name": "a"}}, item: map[string]interface{}{"name": "a"}, kind: reflect.Map, want: 0},
		{name: "nil merge key", order: []interface{}{map[string]interface{}{"name": nil}}, item: map[string]interface{}{"name": nil}, kind: reflect.Map, want: 0},
		{name: "uncomparable query", order: []interface{}{map[string]interface{}{"name": "a"}}, item: map[string]interface{}{"name": []interface{}{1}}, kind: reflect.Map, want: -1},
		{name: "uncomparable order", order: []interface{}{map[string]interface{}{"name": []interface{}{1}}, map[string]interface{}{"name": "a"}}, item: map[string]interface{}{"name": "a"}, kind: reflect.Map, want: 1},
		{name: "nested uncomparable order", order: []interface{}{struct{ Value interface{} }{Value: []int{1}}, "a"}, item: "a", kind: reflect.String, want: 1},
		{name: "nested uncomparable query", order: []interface{}{struct{ Value interface{} }{Value: "a"}}, item: struct{ Value interface{} }{Value: []int{1}}, kind: reflect.Struct, want: -1},
		{name: "uncomparable array order", order: []interface{}{[1]interface{}{[]int{1}}, "a"}, item: "a", kind: reflect.Array, want: 1},
		{name: "uncomparable array query", order: []interface{}{[1]interface{}{"a"}}, item: [1]interface{}{[]int{1}}, kind: reflect.Array, want: -1},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			order := append([]interface{}(nil), tc.order...)
			for len(order) < 32 {
				var item interface{} = fmt.Sprintf("padding-%d", len(order))
				if tc.kind == reflect.Map {
					item = map[string]interface{}{"name": item}
				}
				order = append(order, item)
			}
			lookup := newSliceOrderIndex(order, "name", tc.kind, len(order))
			if got := lookup.index(tc.item); got != tc.want {
				t.Errorf("index = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestSliceOrderingMatchesLinearReference(t *testing.T) {
	for _, kind := range []reflect.Kind{reflect.String, reflect.Map} {
		for _, size := range []int{0, 1, 8, 16, 32, 64, 128} {
			t.Run(fmt.Sprintf("%s/%d", kind, size), func(t *testing.T) {
				for seed := int64(0); seed < 20; seed++ {
					random := rand.New(rand.NewSource(seed))
					order := testOrderItems(size, kind)
					if size > 1 {
						order[size-1] = order[0]
					}
					items := append([]interface{}(nil), order...)
					if kind == reflect.Map {
						for i, item := range items {
							items[i] = map[string]interface{}{"name": item.(map[string]interface{})["name"], "value": i}
						}
						items = append(items, map[string]interface{}{"name": "absent"}, map[string]interface{}{"name": "deleted", "$patch": "delete"})
					} else {
						items = append(items, "absent")
					}
					random.Shuffle(len(items), func(i, j int) { items[i], items[j] = items[j], items[i] })
					want, wantErr := normalizeSliceOrderLinear(append([]interface{}(nil), items...), order, "name", kind)
					got, gotErr := normalizeSliceOrder(append([]interface{}(nil), items...), order, "name", kind)
					if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(gotErr, wantErr) {
						t.Fatalf("seed %d: normalize = %v, %v; want %v, %v", seed, got, gotErr, want, wantErr)
					}
					var left, right []interface{}
					for i, item := range items {
						if i%2 == 0 {
							left = append(left, item)
						} else {
							right = append(right, item)
						}
					}
					want = mergeSortedSliceLinear(left, right, order, "name", kind)
					got = mergeSortedSlice(left, right, order, "name", kind)
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("seed %d: merge = %v, want %v", seed, got, want)
					}
				}
			})
		}
	}
}

func TestStrategicMergePatchListOrderRoundTrip(t *testing.T) {
	for _, size := range []int{8, 32, 128, 1024} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			original := orderTestObject{Items: make([]orderTestItem, size)}
			modified := orderTestObject{}
			for i := range original.Items {
				original.Items[i] = orderTestItem{Name: fmt.Sprintf("item-%04d", i), Value: "before"}
			}
			for i := size - 1; i >= 0; i-- {
				if i%5 == 0 {
					continue
				}
				item := original.Items[i]
				if i%3 == 0 {
					item.Value = "after"
				}
				modified.Items = append(modified.Items, item)
			}
			modified.Items = append(modified.Items, orderTestItem{Name: "new", Value: "added"})
			originalJSON, err := json.Marshal(original)
			if err != nil {
				t.Fatal(err)
			}
			modifiedJSON, err := json.Marshal(modified)
			if err != nil {
				t.Fatal(err)
			}
			patch, err := CreateTwoWayMergePatch(originalJSON, modifiedJSON, orderTestObject{})
			if err != nil {
				t.Fatal(err)
			}
			mergedJSON, err := StrategicMergePatch(originalJSON, patch, orderTestObject{})
			if err != nil {
				t.Fatal(err)
			}
			var merged orderTestObject
			if err := json.Unmarshal(mergedJSON, &merged); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(merged, modified) {
				t.Fatalf("merged = %v, want %v", merged, modified)
			}
		})
	}
}

// Keep the linear lookups as an independent reference for the existing ordering rules.
func normalizeSliceOrderLinear(toSort, order []interface{}, mergeKey string, kind reflect.Kind) ([]interface{}, error) {
	var toDelete []interface{}
	if kind == reflect.Map {
		if err := validateMergeKeyInLists(mergeKey, toSort, order); err != nil {
			return nil, err
		}
		var err error
		toSort, toDelete, err = extractToDeleteItems(toSort)
		if err != nil {
			return nil, err
		}
	}
	sort.SliceStable(toSort, func(i, j int) bool {
		if ii := index(order, toSort[i], mergeKey, kind); ii >= 0 {
			if ij := index(order, toSort[j], mergeKey, kind); ij >= 0 {
				return ii < ij
			}
		}
		return true
	})
	return append(toSort, toDelete...), nil
}

func mergeSortedSliceLinear(left, right, order []interface{}, mergeKey string, kind reflect.Kind) []interface{} {
	result := make([]interface{}, 0, len(left)+len(right))
	i, j := 0, 0
	for i < len(left) && j < len(right) {
		li := index(order, left[i], mergeKey, kind)
		ri := index(order, right[j], mergeKey, kind)
		if li >= 0 && ri >= 0 && li < ri {
			result = append(result, left[i])
			i++
		} else {
			result = append(result, right[j])
			j++
		}
	}
	result = append(result, left[i:]...)
	return append(result, right[j:]...)
}
