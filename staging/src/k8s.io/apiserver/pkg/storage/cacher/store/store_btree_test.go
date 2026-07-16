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

package store

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStoreListOrdered(t *testing.T) {
	store := newThreadedBtreeStoreIndexer(nil, btreeDegree)
	require.NoError(t, store.Add(testStorageElement("foo3", "bar3", 1)))
	require.NoError(t, store.Add(testStorageElement("foo1", "bar2", 2)))
	require.NoError(t, store.Add(testStorageElement("foo2", "bar1", 3)))
	assert.Equal(t, []interface{}{
		testStorageElement("foo1", "bar2", 2),
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, store.List())
}

func TestStoreListPrefix(t *testing.T) {
	store := newThreadedBtreeStoreIndexer(nil, btreeDegree)
	require.NoError(t, store.Add(testStorageElement("foo3", "bar3", 1)))
	require.NoError(t, store.Add(testStorageElement("foo1", "bar2", 2)))
	require.NoError(t, store.Add(testStorageElement("foo2", "bar1", 3)))
	require.NoError(t, store.Add(testStorageElement("bar", "baz", 4)))

	items, err := store.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo1", "bar2", 2),
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items, err = store.OrderedListPrefix("foo2", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo2", "bar1", 3),
	}, items)

	items, err = store.OrderedListPrefix("foo", "foo1\x00")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items, err = store.OrderedListPrefix("foo", "foo2\x00")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items, err = store.OrderedListPrefix("bar", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("bar", "baz", 4),
	}, items)
}

func TestStoreSnapshotter(t *testing.T) {
	cache := NewSnapshotter()
	cache.Add(10, fakeIndexer{rv: 10})
	cache.Add(20, fakeIndexer{rv: 20})
	cache.Add(30, fakeIndexer{rv: 30})
	cache.Add(40, fakeIndexer{rv: 40})
	assert.Equal(t, 4, cache.Len())

	t.Log("No snapshot from before first RV")
	_, found := cache.GetLessOrEqual(9)
	assert.False(t, found)

	t.Log("Get snapshot from first RV")
	snapshot, found := cache.GetLessOrEqual(10)
	assert.True(t, found)
	assert.Equal(t, 10, snapshot.(fakeIndexer).rv)

	t.Log("Get first snapshot by larger RV")
	snapshot, found = cache.GetLessOrEqual(11)
	assert.True(t, found)
	assert.Equal(t, 10, snapshot.(fakeIndexer).rv)

	t.Log("Get second snapshot by larger RV")
	snapshot, found = cache.GetLessOrEqual(22)
	assert.True(t, found)
	assert.Equal(t, 20, snapshot.(fakeIndexer).rv)

	t.Log("Get third snapshot for future revision")
	snapshot, found = cache.GetLessOrEqual(100)
	assert.True(t, found)
	assert.Equal(t, 40, snapshot.(fakeIndexer).rv)

	t.Log("Remove snapshot less than 30")
	cache.RemoveLess(30)

	assert.Equal(t, 2, cache.Len())
	_, found = cache.GetLessOrEqual(10)
	assert.False(t, found)

	_, found = cache.GetLessOrEqual(20)
	assert.False(t, found)

	snapshot, found = cache.GetLessOrEqual(30)
	assert.True(t, found)
	assert.Equal(t, 30, snapshot.(fakeIndexer).rv)

	t.Log("Remove removing all RVs")
	cache.Reset()
	assert.Equal(t, 0, cache.Len())
	_, found = cache.GetLessOrEqual(30)
	assert.False(t, found)
	_, found = cache.GetLessOrEqual(40)
	assert.False(t, found)
}

type fakeIndexer struct {
	rv int
}

func (f fakeIndexer) Add(obj interface{}) error    { return nil }
func (f fakeIndexer) Update(obj interface{}) error { return nil }
func (f fakeIndexer) Delete(obj interface{}) error { return nil }
func (f fakeIndexer) Clone() Snapshot              { return f }
func (f fakeIndexer) OrderedListPrefix(prefixKey, continueKey string) ([]interface{}, error) {
	return nil, nil
}
func (f fakeIndexer) ByIndex(indexName string, indexedValue string) ([]interface{}, error) {
	return nil, nil
}

func (f fakeIndexer) Get(obj interface{}) (item interface{}, exists bool, err error) {
	return nil, false, nil
}

func (f fakeIndexer) GetByKey(key string) (item interface{}, exists bool, err error) {
	return nil, false, nil
}

func (f fakeIndexer) List() []interface{} {
	return nil
}

func (f fakeIndexer) ListKeys() []string {
	return nil
}

func (f fakeIndexer) Replace([]interface{}, string) error {
	return nil
}
func (f fakeIndexer) Count(prefixKey, continueKey string) int { return 0 }

type fakeSnapshotter struct {
	getLessOrEqual func(rv uint64) (Snapshot, bool)
}

var _ Snapshotter = (*fakeSnapshotter)(nil)

func (f *fakeSnapshotter) Reset() {}
func (f *fakeSnapshotter) GetLessOrEqual(rv uint64) (Snapshot, bool) {
	if f.getLessOrEqual == nil {
		return nil, false
	}
	return f.getLessOrEqual(rv)
}
func (f *fakeSnapshotter) Latest() (Snapshot, bool) {
	return nil, false
}
func (f *fakeSnapshotter) Add(rv uint64, indexer Indexer) {}
func (f *fakeSnapshotter) RemoveLess(rv uint64)           {}
func (f *fakeSnapshotter) Len() int {
	return 0
}
