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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/consistency"
)

func TestStoreListOrdered(t *testing.T) {
	store := NewWatchCacheStorage(nil)
	prev, err := store.UpdateStore(watch.Added, testStorageElement("foo3", "bar3", 1), 1)
	require.NoError(t, err)
	assert.Nil(t, prev)
	prev, err = store.UpdateStore(watch.Added, testStorageElement("foo1", "bar2", 2), 2)
	require.NoError(t, err)
	assert.Nil(t, prev)
	prev, err = store.UpdateStore(watch.Added, testStorageElement("foo2", "bar1", 3), 3)
	require.NoError(t, err)
	assert.Nil(t, prev)
	items, err := store.LatestSnapshot().OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo1", "bar2", 2),
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, items)
}

func TestStoreListPrefix(t *testing.T) {
	store := NewWatchCacheStorage(nil)
	prev, err := store.UpdateStore(watch.Added, testStorageElement("foo3", "bar3", 1), 1)
	require.NoError(t, err)
	assert.Nil(t, prev)
	prev, err = store.UpdateStore(watch.Added, testStorageElement("foo1", "bar2", 2), 2)
	require.NoError(t, err)
	assert.Nil(t, prev)
	prev, err = store.UpdateStore(watch.Added, testStorageElement("foo2", "bar1", 3), 3)
	require.NoError(t, err)
	assert.Nil(t, prev)
	prev, err = store.UpdateStore(watch.Added, testStorageElement("bar", "baz", 4), 4)
	require.NoError(t, err)
	assert.Nil(t, prev)

	items, err := store.LatestSnapshot().OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo1", "bar2", 2),
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items, err = store.LatestSnapshot().OrderedListPrefix("foo2", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo2", "bar1", 3),
	}, items)

	items, err = store.LatestSnapshot().OrderedListPrefix("foo", "foo1\x00")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items, err = store.LatestSnapshot().OrderedListPrefix("foo", "foo2\x00")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items, err = store.LatestSnapshot().OrderedListPrefix("bar", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("bar", "baz", 4),
	}, items)
}

func TestStoreSnapshotter(t *testing.T) {
	prevPanic := consistency.PanicOnCacheInconsistency
	consistency.PanicOnCacheInconsistency = true
	t.Cleanup(func() {
		consistency.PanicOnCacheInconsistency = prevPanic
	})

	cache := newSnapshotter(true)
	assert.False(t, cache.HasSnapshot(10))
	_, err := cache.GetSnapshot(10)
	assert.True(t, errors.IsResourceExpired(err))

	snap10 := &btreeStore{tree: newBtreeStore(btreeDegree).tree, resourceVersion: 10}
	snap20 := &btreeStore{tree: newBtreeStore(btreeDegree).tree, resourceVersion: 20}
	snap30 := &btreeStore{tree: newBtreeStore(btreeDegree).tree, resourceVersion: 30}
	snap40 := &btreeStore{tree: newBtreeStore(btreeDegree).tree, resourceVersion: 40}
	cache.Add(snap10)
	cache.Add(snap20)
	cache.Add(snap30)
	cache.Add(snap40)
	assert.Equal(t, 4, cache.Len())

	t.Log("Added snapshot need to have strictly increasing RV")
	assert.Panics(t, func() {
		cache.Add(&btreeStore{resourceVersion: 40})
	})
	assert.Panics(t, func() {
		cache.Add(&btreeStore{resourceVersion: 35})
	})
	t.Log("Bookmarks can have non-decreasing RV")
	cache.UpdateResourceVersion(40)
	assert.Panics(t, func() {
		cache.UpdateResourceVersion(39)
	})
	cache.UpdateResourceVersion(45)

	t.Log("No snapshot from before first RV")
	assert.False(t, cache.HasSnapshot(9))
	_, err = cache.GetSnapshot(9)
	assert.True(t, errors.IsResourceExpired(err))

	t.Log("Get snapshot from first RV")
	assert.True(t, cache.HasSnapshot(10))
	snapshot, err := cache.GetSnapshot(10)
	require.NoError(t, err)
	assert.Equal(t, uint64(10), snapshot.ResourceVersion())
	assert.Same(t, snap10.tree, snapshot.tree)

	t.Log("Get first snapshot by larger RV")
	assert.True(t, cache.HasSnapshot(11))
	snapshot, err = cache.GetSnapshot(11)
	require.NoError(t, err)
	assert.Equal(t, uint64(11), snapshot.ResourceVersion())
	assert.Same(t, snap10.tree, snapshot.tree)

	t.Log("Get second snapshot by larger RV")
	assert.True(t, cache.HasSnapshot(22))
	snapshot, err = cache.GetSnapshot(22)
	require.NoError(t, err)
	assert.Equal(t, uint64(22), snapshot.ResourceVersion())
	assert.Same(t, snap20.tree, snapshot.tree)

	t.Log("Get third snapshot for future revision")
	assert.True(t, cache.HasSnapshot(100))
	assert.Panics(t, func() {
		_, _ = cache.GetSnapshot(100)
	})
	consistency.PanicOnCacheInconsistency = false
	_, err = cache.GetSnapshot(100)
	assert.True(t, storage.IsTooLargeResourceVersion(err))
	consistency.PanicOnCacheInconsistency = true
	cache.UpdateResourceVersion(100)
	snapshot, err = cache.GetSnapshot(100)
	require.NoError(t, err)
	assert.Equal(t, uint64(100), snapshot.ResourceVersion())
	assert.Same(t, snap40.tree, snapshot.tree)

	t.Log("Remove snapshot less than 30")
	cache.RemoveLess(30)

	assert.Equal(t, 2, cache.Len())
	assert.False(t, cache.HasSnapshot(10))
	_, err = cache.GetSnapshot(10)
	assert.True(t, errors.IsResourceExpired(err))

	assert.False(t, cache.HasSnapshot(20))
	_, err = cache.GetSnapshot(20)
	assert.True(t, errors.IsResourceExpired(err))

	assert.True(t, cache.HasSnapshot(30))
	snapshot, err = cache.GetSnapshot(30)
	require.NoError(t, err)
	assert.Equal(t, uint64(30), snapshot.ResourceVersion())
	assert.Same(t, snap30.tree, snapshot.tree)

	t.Log("Replace resets old RVs and adds the new snapshot")
	cache.Replace(&btreeStore{resourceVersion: 200})
	assert.Equal(t, 1, cache.Len())
	assert.False(t, cache.HasSnapshot(30))
	_, err = cache.GetSnapshot(30)
	assert.True(t, errors.IsResourceExpired(err))
	assert.False(t, cache.HasSnapshot(40))
	_, err = cache.GetSnapshot(40)
	assert.True(t, errors.IsResourceExpired(err))
	assert.False(t, cache.HasSnapshot(100))
	_, err = cache.GetSnapshot(100)
	assert.True(t, errors.IsResourceExpired(err))
	assert.True(t, cache.HasSnapshot(200))
	snapshot, err = cache.GetSnapshot(200)
	require.NoError(t, err)
	assert.Equal(t, uint64(200), snapshot.ResourceVersion())
	assert.True(t, cache.HasSnapshot(201))
	assert.Panics(t, func() {
		_, _ = cache.GetSnapshot(201)
	})
	consistency.PanicOnCacheInconsistency = false
	_, err = cache.GetSnapshot(201)
	assert.True(t, storage.IsTooLargeResourceVersion(err))
	consistency.PanicOnCacheInconsistency = true

	t.Log("Disabling snapshotter clears all RVs and ignores updates")
	cache.SetEnabled(false)
	assert.False(t, cache.Enabled())
	assert.Equal(t, 0, cache.Len())
	assert.False(t, cache.HasSnapshot(200))
	_, err = cache.GetSnapshot(200)
	assert.True(t, errors.IsResourceExpired(err))
	cache.Add(&btreeStore{resourceVersion: 300})
	cache.UpdateResourceVersion(400)
	assert.Equal(t, 0, cache.Len())
	assert.False(t, cache.HasSnapshot(300))
	_, err = cache.GetSnapshot(300)
	assert.True(t, errors.IsResourceExpired(err))

	t.Log("Enabling snapshotter clears all RVs")
	cache.SetEnabled(true)
	assert.True(t, cache.Enabled())
	assert.Equal(t, 0, cache.Len())
	assert.False(t, cache.HasSnapshot(300))
	_, err = cache.GetSnapshot(300)
	assert.True(t, errors.IsResourceExpired(err))
	assert.False(t, cache.HasSnapshot(400))
	_, err = cache.GetSnapshot(400)
	assert.True(t, errors.IsResourceExpired(err))

	cache.Add(&btreeStore{resourceVersion: 500})
	assert.Equal(t, 1, cache.Len())
	cache.SetEnabled(true)
	assert.Equal(t, 0, cache.Len())
	assert.False(t, cache.HasSnapshot(500))
	_, err = cache.GetSnapshot(500)
	assert.True(t, errors.IsResourceExpired(err))
}
