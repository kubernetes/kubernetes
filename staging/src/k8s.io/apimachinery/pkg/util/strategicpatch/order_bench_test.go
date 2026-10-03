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
	"reflect"
	"testing"
)

func testOrderItems(size int, kind reflect.Kind) []interface{} {
	items := make([]interface{}, size)
	for i := range items {
		name := fmt.Sprintf("item-%04d", i)
		if kind == reflect.Map {
			items[i] = map[string]interface{}{"name": name}
		} else {
			items[i] = name
		}
	}
	return items
}

func BenchmarkNormalizeSliceOrder(b *testing.B) {
	for _, kind := range []reflect.Kind{reflect.String, reflect.Map} {
		for _, size := range []int{1, 8, 16, 32, 64, 128, 1024} {
			b.Run(fmt.Sprintf("%s/%d", kind, size), func(b *testing.B) {
				order := testOrderItems(size, kind)
				reversed := make([]interface{}, size)
				for i := range order {
					reversed[i] = order[size-1-i]
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := normalizeSliceOrder(append([]interface{}(nil), reversed...), order, "name", kind); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func BenchmarkMergeSortedSlice(b *testing.B) {
	for _, kind := range []reflect.Kind{reflect.String, reflect.Map} {
		for _, size := range []int{1, 8, 16, 32, 64, 128, 1024} {
			b.Run(fmt.Sprintf("%s/%d", kind, size), func(b *testing.B) {
				order := testOrderItems(size, kind)
				var left, right []interface{}
				for i, item := range order {
					if i%2 == 0 {
						left = append(left, item)
					} else {
						right = append(right, item)
					}
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					mergeSortedSlice(left, right, order, "name", kind)
				}
			})
		}
	}
}

type orderTestItem struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type orderTestObject struct {
	Items []orderTestItem `json:"items" patchStrategy:"merge" patchMergeKey:"name"`
}

func BenchmarkCreateTwoWayMergePatchListOrder(b *testing.B) {
	for _, size := range []int{8, 128, 1024} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			original := orderTestObject{Items: make([]orderTestItem, size)}
			modified := orderTestObject{Items: make([]orderTestItem, size)}
			for i := range original.Items {
				original.Items[i] = orderTestItem{Name: fmt.Sprintf("item-%04d", i), Value: "before"}
				modified.Items[size-1-i] = orderTestItem{Name: original.Items[i].Name, Value: "after"}
			}
			originalJSON, err := json.Marshal(original)
			if err != nil {
				b.Fatal(err)
			}
			modifiedJSON, err := json.Marshal(modified)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := CreateTwoWayMergePatch(originalJSON, modifiedJSON, orderTestObject{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
