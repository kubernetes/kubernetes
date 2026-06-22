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

package cacher

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/cacher/store"
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
	s := newWatchCacheStorage(keyFunc, indexers)

	assert.True(t, s.snapshottingEnabled.Load())

	t.Log("New cache collects snapshots")
	elem1 := &store.Element{Key: "foo", Object: &mockObject{key: "foo", val: "100"}}
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

func TestWatchCacheStorageMatchExactResourceVersionFallback(t *testing.T) {
	keyFunc := func(obj runtime.Object) (string, error) {
		return obj.(*mockObject).key, nil
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ListFromCacheSnapshot, true)

	indexers := &cache.Indexers{}
	s := newWatchCacheStorage(keyFunc, indexers)

	t.Log("Initially no snapshots exist, should return ResourceExpired error")
	_, err := s.GetExactSnapshotLocked(20)
	if !errors.IsResourceExpired(err) {
		t.Fatalf("Expected ResourceExpired error, got: %v", err)
	}

	t.Log("Add object at RV 20 to create a snapshot")
	olderElement := &store.Element{Key: "foo", Object: &mockObject{key: "foo", val: "20"}}
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
	if !ok || val.(*store.Element).Object.(*mockObject).val != "20" {
		t.Fatalf("Unexpected element in snapshot")
	}

	t.Log("Add object at RV 30 to create another snapshot")
	newerElement := &store.Element{Key: "foo", Object: &mockObject{key: "foo", val: "30"}}
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
	if !ok || val.(*store.Element).Object.(*mockObject).val != "30" {
		t.Fatalf("Unexpected element in snapshot at RV 30")
	}
}
