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

	"k8s.io/apimachinery/pkg/watch"
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
	cache := newSnapshotter(true)
	cache.Add(&btreeStore{resourceVersion: 10})
	cache.Add(&btreeStore{resourceVersion: 20})
	cache.Add(&btreeStore{resourceVersion: 30})
	cache.Add(&btreeStore{resourceVersion: 40})
	assert.Equal(t, 4, cache.Len())

	t.Log("No snapshot from before first RV")
	_, found := cache.GetLessOrEqual(9)
	assert.False(t, found)

	t.Log("Get snapshot from first RV")
	snapshot, found := cache.GetLessOrEqual(10)
	assert.True(t, found)
	assert.Equal(t, uint64(10), snapshot.ResourceVersion())

	t.Log("Get first snapshot by larger RV")
	snapshot, found = cache.GetLessOrEqual(11)
	assert.True(t, found)
	assert.Equal(t, uint64(10), snapshot.ResourceVersion())

	t.Log("Get second snapshot by larger RV")
	snapshot, found = cache.GetLessOrEqual(22)
	assert.True(t, found)
	assert.Equal(t, uint64(20), snapshot.ResourceVersion())

	t.Log("Get third snapshot for future revision")
	snapshot, found = cache.GetLessOrEqual(100)
	assert.True(t, found)
	assert.Equal(t, uint64(40), snapshot.ResourceVersion())

	t.Log("Remove snapshot less than 30")
	cache.RemoveLess(30)

	assert.Equal(t, 2, cache.Len())
	_, found = cache.GetLessOrEqual(10)
	assert.False(t, found)

	_, found = cache.GetLessOrEqual(20)
	assert.False(t, found)

	snapshot, found = cache.GetLessOrEqual(30)
	assert.True(t, found)
	assert.Equal(t, uint64(30), snapshot.ResourceVersion())

	t.Log("Replace resets old RVs and adds the new snapshot")
	cache.Replace(&btreeStore{resourceVersion: 200})
	assert.Equal(t, 1, cache.Len())
	_, found = cache.GetLessOrEqual(30)
	assert.False(t, found)
	_, found = cache.GetLessOrEqual(40)
	assert.False(t, found)
	_, found = cache.GetLessOrEqual(100)
	assert.False(t, found)
	snapshot, found = cache.GetLessOrEqual(200)
	assert.True(t, found)
	assert.Equal(t, uint64(200), snapshot.ResourceVersion())

	t.Log("Disabling snapshotter clears all RVs and ignores updates")
	cache.SetEnabled(false)
	assert.False(t, cache.Enabled())
	assert.Equal(t, 0, cache.Len())
	_, found = cache.GetLessOrEqual(200)
	assert.False(t, found)
	cache.Add(&btreeStore{resourceVersion: 300})
	cache.UpdateResourceVersion(400)
	assert.Equal(t, 0, cache.Len())
	_, found = cache.GetLessOrEqual(300)
	assert.False(t, found)

	t.Log("Enabling snapshotter clears all RVs")
	cache.SetEnabled(true)
	assert.True(t, cache.Enabled())
	assert.Equal(t, 0, cache.Len())
	_, found = cache.GetLessOrEqual(300)
	assert.False(t, found)
	_, found = cache.GetLessOrEqual(400)
	assert.False(t, found)

	cache.Add(&btreeStore{resourceVersion: 500})
	assert.Equal(t, 1, cache.Len())
	cache.SetEnabled(true)
	assert.Equal(t, 0, cache.Len())
	_, found = cache.GetLessOrEqual(500)
	assert.False(t, found)
}
