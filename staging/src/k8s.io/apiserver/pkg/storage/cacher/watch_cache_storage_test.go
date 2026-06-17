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
