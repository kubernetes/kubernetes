/*
Copyright 2024 The Kubernetes Authors.

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

package cacher

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/storage"
)

func TestStoreListOrdered(t *testing.T) {
	store := newThreadedBtreeStoreIndexer(nil, btreeDegree)
	assert.NoError(t, store.Add(testStorageElement("foo3", "bar3", 1)))
	assert.NoError(t, store.Add(testStorageElement("foo1", "bar2", 2)))
	assert.NoError(t, store.Add(testStorageElement("foo2", "bar1", 3)))
	assert.Equal(t, []interface{}{
		testStorageElement("foo1", "bar2", 2),
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, store.List())
}

func TestStoreListPrefix(t *testing.T) {
	store := newThreadedBtreeStoreIndexer(nil, btreeDegree)
	foo3Field1 := testStorageElementWithFieldsLabels("foo3", "bar3", 1, nil, fields.Set{"key": "value1"})
	foo1Label1 := testStorageElementWithFieldsLabels("foo1", "bar2", 2, labels.Set{"key": "value1"}, nil)
	foo2Label2 := testStorageElementWithFieldsLabels("foo2", "bar1", 3, labels.Set{"key": "value2"}, nil)
	barField2 := testStorageElementWithFieldsLabels("bar", "baz", 4, nil, fields.Set{"key": "value2"})
	bazLabel1Field2 := testStorageElementWithFieldsLabels("baz", "zab", 5, labels.Set{"key": "value1"}, fields.Set{"key": "value2"})

	assert.NoError(t, store.Add(foo3Field1))
	assert.NoError(t, store.Add(foo1Label1))
	assert.NoError(t, store.Add(foo2Label2))
	assert.NoError(t, store.Add(barField2))
	assert.NoError(t, store.Add(bazLabel1Field2))

	tcs := []struct {
		name          string
		prefix        string
		continueKey   string
		limit         int
		predicate     storage.SelectionPredicate
		expectHasMore bool
		expectItems   []interface{}
	}{
		// Without filter
		{
			name:      "Everything",
			predicate: storage.Everything,
			expectItems: []interface{}{
				barField2,
				bazLabel1Field2,
				foo1Label1,
				foo2Label2,
				foo3Field1,
			},
		},
		{
			name:      "Foo2",
			prefix:    "foo2",
			predicate: storage.Everything,
			expectItems: []interface{}{
				foo2Label2,
			},
		},
		{
			name:      "Foo",
			prefix:    "foo",
			predicate: storage.Everything,
			expectItems: []interface{}{
				foo1Label1,
				foo2Label2,
				foo3Field1,
			},
		},
		{
			name:          "Foo Limit",
			prefix:        "foo",
			limit:         1,
			predicate:     storage.Everything,
			expectHasMore: true,
			expectItems: []interface{}{
				foo1Label1,
			},
		},
		{
			name:          "Foo Continue Foo1",
			prefix:        "foo",
			limit:         1,
			continueKey:   "foo1\x00",
			predicate:     storage.Everything,
			expectHasMore: true,
			expectItems: []interface{}{
				foo2Label2,
			},
		},
		{
			name:        "Foo Continue Foo2",
			prefix:      "foo",
			limit:       1,
			continueKey: "foo2\x00",
			predicate:   storage.Everything,
			expectItems: []interface{}{
				foo3Field1,
			},
		},
		{
			name:      "Bar",
			prefix:    "bar",
			predicate: storage.Everything,
			expectItems: []interface{}{
				barField2,
			},
		},
		{
			name:      "Baz",
			prefix:    "baz",
			predicate: storage.Everything,
			expectItems: []interface{}{
				bazLabel1Field2,
			},
		},
		// Filter
		{
			name:      "Field1",
			predicate: storage.SelectionPredicate{Field: fields.SelectorFromSet(fields.Set{"key": "value1"}), Label: labels.Everything()},
			expectItems: []interface{}{
				foo3Field1,
			},
		},
		{
			name:      "Label1",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value1"}), Field: fields.Everything()},
			expectItems: []interface{}{
				bazLabel1Field2, foo1Label1,
			},
		},
		{
			name:      "Field2",
			predicate: storage.SelectionPredicate{Field: fields.SelectorFromSet(fields.Set{"key": "value2"}), Label: labels.Everything()},
			expectItems: []interface{}{
				barField2, bazLabel1Field2,
			},
		},
		{
			name:      "Label2",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value2"}), Field: fields.Everything()},
			expectItems: []interface{}{
				foo2Label2,
			},
		},
		{
			name:      "Label1 Field2",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value1"}), Field: fields.SelectorFromSet(fields.Set{"key": "value2"})},
			expectItems: []interface{}{
				bazLabel1Field2,
			},
		},
		{
			name:      "Label2 Field1",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value2"}), Field: fields.SelectorFromSet(fields.Set{"key": "value1"})},
		},
		// With prefix and filter
		{
			name:      "Foo Field1",
			prefix:    "foo",
			predicate: storage.SelectionPredicate{Field: fields.SelectorFromSet(fields.Set{"key": "value1"}), Label: labels.Everything()},
			expectItems: []interface{}{
				foo3Field1,
			},
		},
		{
			name:      "Foo Label1",
			prefix:    "foo",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value1"}), Field: fields.Everything()},
			expectItems: []interface{}{
				foo1Label1,
			},
		},
		{
			name:      "Foo Label2",
			prefix:    "foo",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value2"}), Field: fields.Everything()},
			expectItems: []interface{}{
				foo2Label2,
			},
		},
		{
			name:      "Foo Label1 Value2",
			prefix:    "foo",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value1"}), Field: fields.SelectorFromSet(fields.Set{"key": "value2"})},
		},
		{
			name:      "Baz Label1 Value2",
			prefix:    "baz",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value1"}), Field: fields.SelectorFromSet(fields.Set{"key": "value2"})},
			expectItems: []interface{}{
				bazLabel1Field2,
			},
		},
		// With limit and filter
		{
			name:      "Limit Field1",
			predicate: storage.SelectionPredicate{Field: fields.SelectorFromSet(fields.Set{"key": "value1"}), Label: labels.Everything()},
			limit:     1,
			expectItems: []interface{}{
				foo3Field1,
			},
		},
		{
			name:          "Limit Label1",
			predicate:     storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value1"}), Field: fields.Everything()},
			limit:         1,
			expectHasMore: true,
			expectItems: []interface{}{
				bazLabel1Field2,
			},
		},
		{
			name:          "Limit Field2",
			predicate:     storage.SelectionPredicate{Field: fields.SelectorFromSet(fields.Set{"key": "value2"}), Label: labels.Everything()},
			limit:         1,
			expectHasMore: true,
			expectItems: []interface{}{
				barField2,
			},
		},
		{
			name:      "Limit Label2",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value2"}), Field: fields.Everything()},
			limit:     1,
			expectItems: []interface{}{
				foo2Label2,
			},
		},
		{
			name:      "Limit Label1 Field2",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value1"}), Field: fields.SelectorFromSet(fields.Set{"key": "value2"})},
			limit:     1,
			expectItems: []interface{}{
				bazLabel1Field2,
			},
		},
		{
			name:      "Limit Label2 Field1",
			predicate: storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value2"}), Field: fields.SelectorFromSet(fields.Set{"key": "value1"})},
			limit:     1,
		},
		// With Continue and filter
		{
			name:        "Continue Label1",
			predicate:   storage.SelectionPredicate{Label: labels.SelectorFromSet(labels.Set{"key": "value1"}), Field: fields.Everything()},
			limit:       1,
			continueKey: "baz\x00",
			expectItems: []interface{}{
				foo1Label1,
			},
		},
		{
			name:        "Limit Field2",
			predicate:   storage.SelectionPredicate{Field: fields.SelectorFromSet(fields.Set{"key": "value2"}), Label: labels.Everything()},
			limit:       1,
			continueKey: "bar\x00",
			expectItems: []interface{}{
				bazLabel1Field2,
			},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			items, hasMore := store.ListPrefix(tc.prefix, tc.continueKey, tc.limit, tc.predicate)
			assert.Equal(t, tc.expectHasMore, hasMore)
			assert.Equal(t, tc.expectItems, items)
		})
	}
}

func TestStoreSnapshotter(t *testing.T) {
	cache := newStoreSnapshotter()
	cache.Add(10, fakeOrderedLister{rv: 10})
	cache.Add(20, fakeOrderedLister{rv: 20})
	cache.Add(30, fakeOrderedLister{rv: 30})
	cache.Add(40, fakeOrderedLister{rv: 40})
	assert.Equal(t, 4, cache.snapshots.Len())

	t.Log("No snapshot from before first RV")
	_, found := cache.GetLessOrEqual(9)
	assert.False(t, found)

	t.Log("Get snapshot from first RV")
	snapshot, found := cache.GetLessOrEqual(10)
	assert.True(t, found)
	assert.Equal(t, 10, snapshot.(fakeOrderedLister).rv)

	t.Log("Get first snapshot by larger RV")
	snapshot, found = cache.GetLessOrEqual(11)
	assert.True(t, found)
	assert.Equal(t, 10, snapshot.(fakeOrderedLister).rv)

	t.Log("Get second snapshot by larger RV")
	snapshot, found = cache.GetLessOrEqual(22)
	assert.True(t, found)
	assert.Equal(t, 20, snapshot.(fakeOrderedLister).rv)

	t.Log("Get third snapshot for future revision")
	snapshot, found = cache.GetLessOrEqual(100)
	assert.True(t, found)
	assert.Equal(t, 40, snapshot.(fakeOrderedLister).rv)

	t.Log("Remove snapshot less than 30")
	cache.RemoveLess(30)

	assert.Equal(t, 2, cache.snapshots.Len())
	_, found = cache.GetLessOrEqual(10)
	assert.False(t, found)

	_, found = cache.GetLessOrEqual(20)
	assert.False(t, found)

	snapshot, found = cache.GetLessOrEqual(30)
	assert.True(t, found)
	assert.Equal(t, 30, snapshot.(fakeOrderedLister).rv)

	t.Log("Remove removing all RVs")
	cache.Reset()
	assert.Equal(t, 0, cache.snapshots.Len())
	_, found = cache.GetLessOrEqual(30)
	assert.False(t, found)
	_, found = cache.GetLessOrEqual(40)
	assert.False(t, found)
}

type fakeOrderedLister struct {
	rv int
}

func (f fakeOrderedLister) Add(obj interface{}) error    { return nil }
func (f fakeOrderedLister) Update(obj interface{}) error { return nil }
func (f fakeOrderedLister) Delete(obj interface{}) error { return nil }
func (f fakeOrderedLister) Clone() orderedLister         { return f }
func (f fakeOrderedLister) ListPrefix(prefixKey, continueKey string, limit int, pred storage.SelectionPredicate) ([]interface{}, bool) {
	return nil, false
}
func (f fakeOrderedLister) Count(prefixKey, continueKey string) int { return 0 }
