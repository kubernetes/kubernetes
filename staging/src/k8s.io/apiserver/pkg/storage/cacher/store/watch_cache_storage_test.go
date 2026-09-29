/*
Copyright The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestWatchCacheStorageMarkConsistent(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)

	indexers := &cache.Indexers{}
	s := NewWatchCacheStorage(indexers)

	assert.True(t, s.snapshottingEnabled)

	t.Log("New cache collects snapshots")
	elem1 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "100"}}
	prev, err := s.UpdateStore(watch.Added, elem1, 100)
	require.NoError(t, err)
	assert.Nil(t, prev)
	assert.Equal(t, 1, s.snapshots.Len())
	_, err = s.GetExactSnapshotLocked(100)
	require.NoError(t, err)

	t.Log("Inconsistent cache clears old snapshots")
	s.MarkConsistent(false)
	assert.Equal(t, 0, s.snapshots.Len())
	assert.False(t, s.snapshottingEnabled)
	_, err = s.GetExactSnapshotLocked(100)
	require.Error(t, err)

	t.Log("Inconsistent cache doesn't collect new snapshot")
	prev, err = s.UpdateStore(watch.Modified, elem1, 200)
	require.NoError(t, err)
	assert.Equal(t, elem1, prev)
	assert.Equal(t, 0, s.snapshots.Len())
	_, err = s.GetExactSnapshotLocked(200)
	require.Error(t, err)

	t.Log("Marking cache consistent allows it to collect new snapshots, list skips etcd")
	s.MarkConsistent(true)
	prev, err = s.UpdateStore(watch.Modified, elem1, 300)
	require.NoError(t, err)
	assert.Equal(t, elem1, prev)
	assert.Equal(t, 1, s.snapshots.Len())
	_, err = s.GetExactSnapshotLocked(300)
	require.NoError(t, err)
}

func TestLatestSnapshot(t *testing.T) {
	// Latest snapshot is maintained even with snapshotting disabled.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, false)

	indexers := &cache.Indexers{}
	s := NewWatchCacheStorage(indexers)

	before := s.LatestSnapshot()
	assert.Equal(t, uint64(0), before.ResourceVersion())
	items, err := before.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Empty(t, items, "expected empty snapshot before any writes")

	elem := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "100"}}
	prev, err := s.UpdateStore(watch.Added, elem, 100)
	require.NoError(t, err)
	assert.Nil(t, prev)

	snap := s.LatestSnapshot()
	assert.Equal(t, uint64(100), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, items, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "100"}, items[0].(*Element).Object)

	items, err = before.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Empty(t, items, "snapshot taken before the write must not change")
	assert.Equal(t, uint64(0), before.ResourceVersion(), "snapshot taken before the write must not change")

	s.UpdateResourceVersion(150)
	snap = s.LatestSnapshot()
	assert.Equal(t, uint64(150), snap.ResourceVersion(), "resourceVersion update must advance the latest snapshot")
	items, err = snap.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, items, 1, "resourceVersion update must not change content")

	require.NoError(t, s.Replace(nil, 200))
	snap = s.LatestSnapshot()
	assert.Equal(t, uint64(200), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Empty(t, items)
}

func TestWatchCacheStorageMatchExactResourceVersionFallback(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)

	indexers := &cache.Indexers{}
	s := NewWatchCacheStorage(indexers)

	t.Log("Initially no snapshots exist, should return ResourceExpired error")
	_, err := s.GetExactSnapshotLocked(20)
	if !errors.IsResourceExpired(err) {
		t.Fatalf("Expected ResourceExpired error, got: %v", err)
	}

	t.Log("Add object at RV 20 to create a snapshot")
	olderElement := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "20"}}
	_, err = s.UpdateStore(watch.Added, olderElement, 20)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	snap, err := s.GetExactSnapshotLocked(20)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	val, ok, err := snap.GetByKey("foo")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !ok || val.(*Element).Object.(*mockObject).val != "20" {
		t.Fatalf("Unexpected element in snapshot")
	}

	t.Log("Add object at RV 30 to create another snapshot")
	newerElement := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "30"}}
	_, err = s.UpdateStore(watch.Modified, newerElement, 30)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	t.Log("Compact snapshots up to 30. This deletes snapshot at 20")
	s.Compact(30)

	t.Log("Get snapshot at RV 20 should now return ResourceExpired error")
	_, err = s.GetExactSnapshotLocked(20)
	if !errors.IsResourceExpired(err) {
		t.Fatalf("Expected ResourceExpired error, got: %v", err)
	}

	t.Log("Get snapshot at RV 30 should succeed since it was not compacted")
	snap30, err := s.GetExactSnapshotLocked(30)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	val, ok, err = snap30.GetByKey("foo")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !ok || val.(*Element).Object.(*mockObject).val != "30" {
		t.Fatalf("Unexpected element in snapshot at RV 30")
	}
}

type mockObject struct {
	runtime.Object
	key string
	val string
}

func (m *mockObject) DeepCopyObject() runtime.Object {
	return &mockObject{key: m.key, val: m.val}
}

func TestWatchCacheStorageSnapshots(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)

	indexers := &cache.Indexers{}
	s := NewWatchCacheStorage(indexers)

	assert.True(t, s.snapshottingEnabled, "Expected snapshotting to be enabled when feature gate is active")

	_, err := s.GetExactSnapshotLocked(100)
	require.Error(t, err, "Expected empty cache to not include any snapshots")

	t.Log("Test cache on rev 100")
	elem1 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "100"}}
	prev, err := s.UpdateStore(watch.Added, elem1, 100)
	require.NoError(t, err)
	assert.Nil(t, prev)

	elem2 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "200"}}
	prev, err = s.UpdateStore(watch.Modified, elem2, 200)
	require.NoError(t, err)
	assert.Equal(t, elem1, prev)

	elem3 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "300"}}
	prev, err = s.UpdateStore(watch.Deleted, elem3, 300)
	require.NoError(t, err)
	assert.Equal(t, elem2, prev)

	t.Log("Test cache on rev 100")
	_, err = s.GetExactSnapshotLocked(99)
	require.Error(t, err, "Expected store to not include rev 99")

	snap100, err := s.GetExactSnapshotLocked(100)
	require.NoError(t, err)
	elements, err := snap100.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, elements, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "100"}, elements[0].(*Element).Object)

	t.Log("Compact snapshots to remove rev 100")
	s.CompactSnapshotsLocked(200)
	_, err = s.GetExactSnapshotLocked(100)
	require.Error(t, err, "Expected compacted snapshot at 100 to be deleted")

	t.Log("Test cache on rev 200")
	snap200, err := s.GetExactSnapshotLocked(200)
	require.NoError(t, err)
	elements, err = snap200.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, elements, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "200"}, elements[0].(*Element).Object)

	t.Log("Test cache on rev 300")
	snap300, err := s.GetExactSnapshotLocked(300)
	require.NoError(t, err)
	elements, err = snap300.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Empty(t, elements)

	t.Log("Test cache on rev 400")
	elem4 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "400"}}
	prev, err = s.UpdateStore(watch.Added, elem4, 400)
	require.NoError(t, err)
	assert.Nil(t, prev, "the key was deleted at rev 300, so this add replaces nothing")

	snap400, err := s.GetExactSnapshotLocked(400)
	require.NoError(t, err)
	elements, err = snap400.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, elements, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "400"}, elements[0].(*Element).Object)

	t.Log("Compact snapshots to simulate cache capacity downsize")
	s.CompactSnapshotsLocked(500)
	_, err = s.GetExactSnapshotLocked(499)
	require.Error(t, err, "Expected compacted snapshots below 500 to be deleted")

	t.Log("Test cache on rev 500")
	elem5 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "500"}}
	prev, err = s.UpdateStore(watch.Modified, elem5, 500)
	require.NoError(t, err)
	assert.Equal(t, elem4, prev)

	snap500, err := s.GetExactSnapshotLocked(500)
	require.NoError(t, err)
	elements, err = snap500.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, elements, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "500"}, elements[0].(*Element).Object)

	t.Log("Test cache on rev 600")
	elem6 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "600"}}
	prev, err = s.UpdateStore(watch.Modified, elem6, 600)
	require.NoError(t, err)
	assert.Equal(t, elem5, prev)

	snap600, err := s.GetExactSnapshotLocked(600)
	require.NoError(t, err)
	elements, err = snap600.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, elements, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "600"}, elements[0].(*Element).Object)

	t.Log("Replace cache to remove history")
	_, err = s.GetExactSnapshotLocked(500)
	require.NoError(t, err, "Confirm that cache stores history before replace")

	err = s.Replace([]*Element{
		{Key: "foo", Object: &mockObject{key: "foo", val: "600"}},
	}, 700)
	require.NoError(t, err)

	_, err = s.GetExactSnapshotLocked(500)
	require.Error(t, err, "Expected replace to remove history")
	_, err = s.GetExactSnapshotLocked(600)
	require.Error(t, err, "Expected replace to remove history")

	t.Log("Test cache on rev 700")
	snap700, err := s.GetExactSnapshotLocked(700)
	require.NoError(t, err)
	elements, err = snap700.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, elements, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "600"}, elements[0].(*Element).Object)
}

func TestStoreSingleKey(t *testing.T) {
	store := NewWatchCacheStorage(testStoreIndexers())
	assertStoreEmpty(t, store, "foo", 0)

	prev, err := store.UpdateStore(watch.Added, testStorageElement("foo", "bar", 1), 1)
	require.NoError(t, err)
	assert.Nil(t, prev, "adding a new key replaces nothing")
	assertStoreSingleKey(t, store, "foo", "bar", 1)

	prev, err = store.UpdateStore(watch.Modified, testStorageElement("foo", "baz", 2), 2)
	require.NoError(t, err)
	assert.Equal(t, testStorageElement("foo", "bar", 1), prev)
	assertStoreSingleKey(t, store, "foo", "baz", 2)

	prev, err = store.UpdateStore(watch.Modified, testStorageElement("foo", "baz", 3), 3)
	require.NoError(t, err)
	assert.Equal(t, testStorageElement("foo", "baz", 2), prev)
	assertStoreSingleKey(t, store, "foo", "baz", 3)

	require.NoError(t, store.Replace([]*Element{testStorageElement("foo", "bar", 4)}, 4))
	assertStoreSingleKey(t, store, "foo", "bar", 4)

	prev, err = store.UpdateStore(watch.Deleted, testStorageElement("foo", "", 0), 5)
	require.NoError(t, err)
	assert.Equal(t, testStorageElement("foo", "bar", 4), prev)
	assertStoreEmpty(t, store, "foo", 5)

	prev, err = store.UpdateStore(watch.Deleted, testStorageElement("foo", "", 0), 6)
	require.NoError(t, err)
	assert.Nil(t, prev, "deleting a missing key removes nothing")
	assertStoreEmpty(t, store, "foo", 6)

	store.UpdateResourceVersion(7)
	assertStoreEmpty(t, store, "foo", 7)

	require.NoError(t, store.Replace(nil, 8))
	assertStoreEmpty(t, store, "foo", 8)
}

func TestStoreIndexerSingleKey(t *testing.T) {
	store := NewWatchCacheStorage(testStoreIndexers())
	snap, err := store.GetByIndexSnapshot("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, uint64(0), snap.ResourceVersion())
	items, err := snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Empty(t, items)

	prev, err := store.UpdateStore(watch.Added, testStorageElement("foo", "bar", 1), 1)
	require.NoError(t, err)
	assert.Nil(t, prev)
	snap, err = store.GetByIndexSnapshot("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, uint64(1), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo", "bar", 1),
	}, items)

	prev, err = store.UpdateStore(watch.Modified, testStorageElement("foo", "baz", 2), 2)
	require.NoError(t, err)
	assert.Equal(t, testStorageElement("foo", "bar", 1), prev)
	snap, err = store.GetByIndexSnapshot("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, uint64(2), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Empty(t, items)
	snap, err = store.GetByIndexSnapshot("by_val", "baz")
	require.NoError(t, err)
	assert.Equal(t, uint64(2), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo", "baz", 2),
	}, items)

	prev, err = store.UpdateStore(watch.Modified, testStorageElement("foo", "baz", 3), 3)
	require.NoError(t, err)
	assert.Equal(t, testStorageElement("foo", "baz", 2), prev)
	snap, err = store.GetByIndexSnapshot("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, uint64(3), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Empty(t, items)
	snap, err = store.GetByIndexSnapshot("by_val", "baz")
	require.NoError(t, err)
	assert.Equal(t, uint64(3), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo", "baz", 3),
	}, items)

	require.NoError(t, store.Replace([]*Element{
		testStorageElement("foo", "bar", 4),
	}, 4))
	snap, err = store.GetByIndexSnapshot("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, uint64(4), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{
		testStorageElement("foo", "bar", 4),
	}, items)
	snap, err = store.GetByIndexSnapshot("by_val", "baz")
	require.NoError(t, err)
	assert.Equal(t, uint64(4), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Empty(t, items)

	prev, err = store.UpdateStore(watch.Deleted, testStorageElement("foo", "", 0), 5)
	require.NoError(t, err)
	assert.Equal(t, testStorageElement("foo", "bar", 4), prev)
	snap, err = store.GetByIndexSnapshot("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, uint64(5), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Empty(t, items)
	snap, err = store.GetByIndexSnapshot("by_val", "baz")
	require.NoError(t, err)
	assert.Equal(t, uint64(5), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Empty(t, items)

	prev, err = store.UpdateStore(watch.Deleted, testStorageElement("foo", "", 0), 6)
	require.NoError(t, err)
	assert.Nil(t, prev)
	snap, err = store.GetByIndexSnapshot("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, uint64(6), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Empty(t, items)

	store.UpdateResourceVersion(7)
	snap, err = store.GetByIndexSnapshot("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, uint64(7), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Empty(t, items)

	require.NoError(t, store.Replace(nil, 8))
	snap, err = store.GetByIndexSnapshot("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, uint64(8), snap.ResourceVersion())
	items, err = snap.OrderedListPrefix("foo", "")
	require.NoError(t, err)
	assert.Empty(t, items)
}

func assertStoreEmpty(t *testing.T, store *WatchCacheStorage, nonExistingKey string, expectRV uint64) {
	snap := store.LatestSnapshot()
	assert.Equal(t, expectRV, snap.ResourceVersion())
	item, ok, err := snap.GetByKey(nonExistingKey)
	require.NoError(t, err)
	assert.False(t, ok)
	assert.Nil(t, item)

	items, err := snap.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Empty(t, items)
}

func assertStoreSingleKey(t *testing.T, store *WatchCacheStorage, expectKey, expectValue string, expectRV int) {
	snap := store.LatestSnapshot()
	assert.Equal(t, uint64(expectRV), snap.ResourceVersion())
	item, ok, err := snap.GetByKey(expectKey)
	require.NoError(t, err)
	assert.True(t, ok)
	assert.Equal(t, expectValue, item.(*Element).Object.(fakeObj).value)

	items, err := snap.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Equal(t, []interface{}{testStorageElement(expectKey, expectValue, expectRV)}, items)
}

func testStorageElement(key, value string, rv int) *Element {
	return &Element{Key: key, Object: fakeObj{value: value, rv: rv}}
}

type fakeObj struct {
	value string
	rv    int
}

func (f fakeObj) GetObjectKind() schema.ObjectKind { return nil }
func (f fakeObj) DeepCopyObject() runtime.Object   { return nil }

var _ runtime.Object = (*fakeObj)(nil)

func testStoreIndexFunc(obj interface{}) ([]string, error) {
	return []string{obj.(fakeObj).value}, nil
}

func testStoreIndexers() *cache.Indexers {
	indexers := cache.Indexers{}
	indexers["by_val"] = testStoreIndexFunc
	return &indexers
}
