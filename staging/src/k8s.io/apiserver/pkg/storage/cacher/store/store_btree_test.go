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
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
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
	assert.NoError(t, store.Add(testStorageElement("foo3", "bar3", 1)))
	assert.NoError(t, store.Add(testStorageElement("foo1", "bar2", 2)))
	assert.NoError(t, store.Add(testStorageElement("foo2", "bar1", 3)))
	assert.NoError(t, store.Add(testStorageElement("bar", "baz", 4)))

	items := store.ListPrefix("foo", "")
	assert.Equal(t, []interface{}{
		testStorageElement("foo1", "bar2", 2),
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items = store.ListPrefix("foo2", "")
	assert.Equal(t, []interface{}{
		testStorageElement("foo2", "bar1", 3),
	}, items)

	items = store.ListPrefix("foo", "foo1\x00")
	assert.Equal(t, []interface{}{
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items = store.ListPrefix("foo", "foo2\x00")
	assert.Equal(t, []interface{}{
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items = store.ListPrefix("bar", "")
	assert.Equal(t, []interface{}{
		testStorageElement("bar", "baz", 4),
	}, items)
}

func TestStoreSnapshotter(t *testing.T) {
	cache := NewSnapshotter()
	cache.Add(10, fakeOrderedLister{rv: 10})
	cache.Add(20, fakeOrderedLister{rv: 20})
	cache.Add(30, fakeOrderedLister{rv: 30})
	cache.Add(40, fakeOrderedLister{rv: 40})
	assert.Equal(t, 4, cache.Len())

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

	assert.Equal(t, 2, cache.Len())
	_, found = cache.GetLessOrEqual(10)
	assert.False(t, found)

	_, found = cache.GetLessOrEqual(20)
	assert.False(t, found)

	snapshot, found = cache.GetLessOrEqual(30)
	assert.True(t, found)
	assert.Equal(t, 30, snapshot.(fakeOrderedLister).rv)

	t.Log("Remove removing all RVs")
	cache.Reset()
	assert.Equal(t, 0, cache.Len())
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
func (f fakeOrderedLister) Clone() OrderedLister         { return f }
func (f fakeOrderedLister) ListPrefix(prefixKey, continueKey string) []interface{} {
	return nil
}
func (f fakeOrderedLister) Count(prefixKey, continueKey string) int { return 0 }

type fakeSnapshotter struct {
	getLessOrEqual func(rv uint64) (OrderedLister, bool)
}

var _ Snapshotter = (*fakeSnapshotter)(nil)

func (f *fakeSnapshotter) Reset() {}
func (f *fakeSnapshotter) GetLessOrEqual(rv uint64) (OrderedLister, bool) {
	if f.getLessOrEqual == nil {
		return nil, false
	}
	return f.getLessOrEqual(rv)
}
func (f *fakeSnapshotter) Add(rv uint64, indexer OrderedLister) {}
func (f *fakeSnapshotter) RemoveLess(rv uint64)                 {}
func (f *fakeSnapshotter) Len() int {
	return 0
}

func TestSliceSnapshotterCompaction(t *testing.T) {
	cache := NewSnapshotter()
	// Add 100 entries.
	for i := uint64(1); i <= 100; i++ {
		cache.Add(i, fakeOrderedLister{rv: int(i)})
	}
	assert.Equal(t, 100, cache.Len())

	// Remove the first 80 entries (rv < 81).
	cache.RemoveLess(81)
	assert.Equal(t, 20, cache.Len())

	// Verify that lookups still work correctly after compaction.
	_, found := cache.GetLessOrEqual(80)
	assert.False(t, found)

	snapshot, found := cache.GetLessOrEqual(81)
	assert.True(t, found)
	assert.Equal(t, 81, snapshot.(fakeOrderedLister).rv)

	snapshot, found = cache.GetLessOrEqual(100)
	assert.True(t, found)
	assert.Equal(t, 100, snapshot.(fakeOrderedLister).rv)

	snapshot, found = cache.GetLessOrEqual(200)
	assert.True(t, found)
	assert.Equal(t, 100, snapshot.(fakeOrderedLister).rv)
}

func TestSliceSnapshotterEdgeCases(t *testing.T) {
	cache := NewSnapshotter()

	t.Log("Empty snapshotter: GetLessOrEqual returns not found")
	_, found := cache.GetLessOrEqual(10)
	assert.False(t, found)

	t.Log("Empty snapshotter: RemoveLess is a no-op")
	cache.RemoveLess(10) // should not panic
	assert.Equal(t, 0, cache.Len())

	t.Log("Single element")
	cache.Add(5, fakeOrderedLister{rv: 5})
	assert.Equal(t, 1, cache.Len())

	_, found = cache.GetLessOrEqual(4)
	assert.False(t, found)

	snapshot, found := cache.GetLessOrEqual(5)
	assert.True(t, found)
	assert.Equal(t, 5, snapshot.(fakeOrderedLister).rv)

	snapshot, found = cache.GetLessOrEqual(100)
	assert.True(t, found)
	assert.Equal(t, 5, snapshot.(fakeOrderedLister).rv)

	t.Log("RemoveLess with rv beyond all entries removes everything")
	cache.RemoveLess(1000)
	assert.Equal(t, 0, cache.Len())
	_, found = cache.GetLessOrEqual(5)
	assert.False(t, found)

	t.Log("Reset on empty snapshotter is a no-op")
	cache.Reset()
	assert.Equal(t, 0, cache.Len())

	t.Log("GetLessOrEqual on exact boundary")
	cache.Add(10, fakeOrderedLister{rv: 10})
	cache.Add(20, fakeOrderedLister{rv: 20})
	snapshot, found = cache.GetLessOrEqual(10)
	assert.True(t, found)
	assert.Equal(t, 10, snapshot.(fakeOrderedLister).rv)

	snapshot, found = cache.GetLessOrEqual(20)
	assert.True(t, found)
	assert.Equal(t, 20, snapshot.(fakeOrderedLister).rv)
}

func TestSliceSnapshotterConcurrent(t *testing.T) {
	cache := NewSnapshotter()
	const goroutines = 10
	const ops = 1000

	var wg sync.WaitGroup
	wg.Add(goroutines * 3)

	// Writers
	for g := 0; g < goroutines; g++ {
		go func(offset int) {
			defer wg.Done()
			for i := 0; i < ops; i++ {
				cache.Add(uint64(offset*ops+i), fakeOrderedLister{rv: offset*ops + i})
			}
		}(g)
	}

	// Readers
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < ops; i++ {
				cache.GetLessOrEqual(uint64(i))
			}
		}()
	}

	// Removers
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < ops; i++ {
				cache.RemoveLess(uint64(i))
			}
		}()
	}

	wg.Wait()
}
