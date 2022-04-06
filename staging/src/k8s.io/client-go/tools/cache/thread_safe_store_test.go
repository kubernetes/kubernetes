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

package cache

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/selection"
)

func TestThreadSafeStoreDeleteRemovesEmptySetsFromIndex(t *testing.T) {
	testIndexer := "testIndexer"

	indexers := Indexers{
		testIndexer: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)}
			return indexes, nil
		},
	}

	indices := Indices{}
	store := NewThreadSafeStore(indexers, indices).(*threadSafeMap)

	testKey := "testKey"

	store.Add(testKey, testKey)

	// Assumption check, there should be a set for the `testKey` with one element in the added index
	set := store.indices[testIndexer][testKey]

	if len(set) != 1 {
		t.Errorf("Initial assumption of index backing string set having 1 element failed. Actual elements: %d", len(set))
		return
	}

	store.Delete(testKey)
	set, present := store.indices[testIndexer][testKey]

	if present {
		t.Errorf("Index backing string set not deleted from index. Set length: %d", len(set))
	}
}

func TestThreadSafeStoreAddKeepsNonEmptySetPostDeleteFromIndex(t *testing.T) {
	testIndexer := "testIndexer"
	testIndex := "testIndex"

	indexers := Indexers{
		testIndexer: func(obj interface{}) (strings []string, e error) {
			indexes := []string{testIndex}
			return indexes, nil
		},
	}

	indices := Indices{}
	store := NewThreadSafeStore(indexers, indices).(*threadSafeMap)

	store.Add("retain", "retain")
	store.Add("delete", "delete")

	// Assumption check, there should be a set for the `testIndex` with two elements
	set := store.indices[testIndexer][testIndex]

	if len(set) != 2 {
		t.Errorf("Initial assumption of index backing string set having 2 elements failed. Actual elements: %d", len(set))
		return
	}

	store.Delete("delete")
	set, present := store.indices[testIndexer][testIndex]

	if !present {
		t.Errorf("Index backing string set erroneously deleted from index.")
		return
	}

	if len(set) != 1 {
		t.Errorf("Index backing string set has incorrect length, expect 1. Set length: %d", len(set))
	}
}

func TestThreadSafeStoreByIndexes(t *testing.T) {
	testIndexer1 := "test1Indexer"
	testIndexer2 := "test2Indexer"

	indexers := Indexers{
		testIndexer1: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)[:4]}
			return indexes, nil
		},
		testIndexer2: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)[4:5]}
			return indexes, nil
		},
	}

	indices := Indices{}
	store := NewThreadSafeStore(indexers, indices).(*threadSafeMap)

	store.Add(testIndexer1, testIndexer1)
	store.Add(testIndexer2, testIndexer2)

	tests := []struct {
		name string
		in   IndexConditions
		out  []interface{}
	}{
		{
			name: "equal match",
			in: IndexConditions{
				{Operator: selection.Equals, IndexName: testIndexer1, IndexKey: "test"},
				{Operator: selection.Equals, IndexName: testIndexer2, IndexKey: "1"},
			},
			out: []interface{}{testIndexer1},
		},
		{
			name: "double equal match",
			in: IndexConditions{
				{Operator: selection.DoubleEquals, IndexName: testIndexer1, IndexKey: "test"},
				{Operator: selection.DoubleEquals, IndexName: testIndexer2, IndexKey: "1"},
			},
			out: []interface{}{testIndexer1},
		},
		{
			name: "not equal match",
			in: IndexConditions{
				{Operator: selection.Equals, IndexName: testIndexer1, IndexKey: "test"},
				{Operator: selection.NotEquals, IndexName: testIndexer2, IndexKey: "1"},
			},
			out: []interface{}{testIndexer2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := store.ByIndexes(tt.in)
			if err != nil {
				t.Errorf("ByIndexes() error = %v", err)
				return
			}
			if got[0].(string) != tt.out[0].(string) {
				t.Errorf("ByIndexes() got = %v, want %v", got, tt.out)
			}
		})
	}
}

func BenchmarkIndexer(b *testing.B) {
	testIndexer := "testIndexer"

	indexers := Indexers{
		testIndexer: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)}
			return indexes, nil
		},
	}

	indices := Indices{}
	store := NewThreadSafeStore(indexers, indices).(*threadSafeMap)

	// The following benchmark imitates what is happening in indexes
	// used in storage layer, where indexing is mostly static (e.g.
	// indexing objects by their (namespace, name)).
	// The 5000 number imitates indexing nodes in 5000-node cluster.
	objectCount := 5000
	objects := make([]string, 0, 5000)
	for i := 0; i < objectCount; i++ {
		objects = append(objects, fmt.Sprintf("object-number-%d", i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.Update(objects[i%objectCount], objects[i%objectCount])
	}
}
