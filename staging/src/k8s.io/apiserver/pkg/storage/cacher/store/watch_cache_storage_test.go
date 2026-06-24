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
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestWatchCacheStorageMarkConsistent(t *testing.T) {
	keyFunc := func(obj runtime.Object) (string, error) {
		return obj.(*mockObject).key, nil
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)

	indexers := &cache.Indexers{}
	s := NewWatchCacheStorage(keyFunc, indexers)

	assert.True(t, s.snapshottingEnabled.Load())

	t.Log("New cache collects snapshots")
	elem1 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "100"}}
	require.NoError(t, s.UpdateStoreLocked(watch.Added, elem1, 100))
	assert.Equal(t, 1, s.snapshots.Len())
	_, err := s.GetExactSnapshotLocked(100)
	require.NoError(t, err)

	t.Log("Inconsistent cache clears old snapshots")
	s.MarkConsistent(false)
	assert.Equal(t, 0, s.snapshots.Len())
	assert.False(t, s.snapshottingEnabled.Load())
	_, err = s.GetExactSnapshotLocked(100)
	require.Error(t, err)

	t.Log("Inconsistent cache doesn't collect new snapshot")
	require.NoError(t, s.UpdateStoreLocked(watch.Modified, elem1, 200))
	assert.Equal(t, 0, s.snapshots.Len())
	_, err = s.GetExactSnapshotLocked(200)
	require.Error(t, err)

	t.Log("Marking cache consistent allows it to collect new snapshots, list skips etcd")
	s.MarkConsistent(true)
	require.NoError(t, s.UpdateStoreLocked(watch.Modified, elem1, 300))
	assert.Equal(t, 1, s.snapshots.Len())
	_, err = s.GetExactSnapshotLocked(300)
	require.NoError(t, err)
}

func TestLatestSnapshotLocked(t *testing.T) {
	keyFunc := func(obj runtime.Object) (string, error) {
		return obj.(*mockObject).key, nil
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)

	indexers := &cache.Indexers{}
	s := NewWatchCacheStorage(keyFunc, indexers)

	_, ok := s.LatestSnapshotLocked()
	assert.False(t, ok, "expected no snapshot before any writes")

	elem := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "100"}}
	require.NoError(t, s.UpdateStoreLocked(watch.Added, elem, 100))

	snap, ok := s.LatestSnapshotLocked()
	require.True(t, ok, "expected snapshot after write")
	items, err := snap.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, items, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "100"}, items[0].(*Element).Object)
}

func TestWatchCacheStorageMatchExactResourceVersionFallback(t *testing.T) {
	keyFunc := func(obj runtime.Object) (string, error) {
		return obj.(*mockObject).key, nil
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)

	indexers := &cache.Indexers{}
	s := NewWatchCacheStorage(keyFunc, indexers)

	t.Log("Initially no snapshots exist, should return ResourceExpired error")
	_, err := s.GetExactSnapshotLocked(20)
	if !errors.IsResourceExpired(err) {
		t.Fatalf("Expected ResourceExpired error, got: %v", err)
	}

	t.Log("Add object at RV 20 to create a snapshot")
	olderElement := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "20"}}
	err = s.UpdateStoreLocked(watch.Added, olderElement, 20)
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
	err = s.UpdateStoreLocked(watch.Modified, newerElement, 30)
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

	keyFunc := func(obj runtime.Object) (string, error) {
		return obj.(*mockObject).key, nil
	}

	indexers := &cache.Indexers{}
	s := NewWatchCacheStorage(keyFunc, indexers)

	assert.True(t, s.snapshottingEnabled.Load(), "Expected snapshotting to be enabled when feature gate is active")

	_, err := s.GetExactSnapshotLocked(100)
	require.Error(t, err, "Expected empty cache to not include any snapshots")

	t.Log("Test cache on rev 100")
	elem1 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "100"}}
	require.NoError(t, s.UpdateStoreLocked(watch.Added, elem1, 100))

	elem2 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "200"}}
	require.NoError(t, s.UpdateStoreLocked(watch.Modified, elem2, 200))

	elem3 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "300"}}
	require.NoError(t, s.UpdateStoreLocked(watch.Deleted, elem3, 300))

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
	require.NoError(t, s.UpdateStoreLocked(watch.Added, elem4, 400))

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
	require.NoError(t, s.UpdateStoreLocked(watch.Modified, elem5, 500))

	snap500, err := s.GetExactSnapshotLocked(500)
	require.NoError(t, err)
	elements, err = snap500.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, elements, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "500"}, elements[0].(*Element).Object)

	t.Log("Test cache on rev 600")
	elem6 := &Element{Key: "foo", Object: &mockObject{key: "foo", val: "600"}}
	require.NoError(t, s.UpdateStoreLocked(watch.Modified, elem6, 600))

	snap600, err := s.GetExactSnapshotLocked(600)
	require.NoError(t, err)
	elements, err = snap600.OrderedListPrefix("", "")
	require.NoError(t, err)
	assert.Len(t, elements, 1)
	assert.Equal(t, &mockObject{key: "foo", val: "600"}, elements[0].(*Element).Object)

	t.Log("Replace cache to remove history")
	_, err = s.GetExactSnapshotLocked(500)
	require.NoError(t, err, "Confirm that cache stores history before replace")

	err = s.ReplaceLocked([]interface{}{
		&Element{Key: "foo", Object: &mockObject{key: "foo", val: "600"}},
	}, "700", 700)
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
