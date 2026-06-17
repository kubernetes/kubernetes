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
	"fmt"
	"sort"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/cacher/key"
	"k8s.io/apiserver/pkg/storage/cacher/store"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
)

func newWatchCacheStorage(keyFunc func(runtime.Object) (string, error), indexers *cache.Indexers) *watchCacheStorage {
	storage := &watchCacheStorage{
		keyFunc:             keyFunc,
		store:               store.NewIndexer(indexers),
		listResourceVersion: 0,
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
		storage.snapshottingEnabled.Store(true)
		storage.snapshots = store.NewSnapshotter()
	}
	return storage
}

type watchCacheStorage struct {
	keyFunc func(runtime.Object) (string, error)

	// store will effectively support LIST operation from the "end of cache
	// history" i.e. from the moment just after the newest cached watched event.
	// It is necessary to effectively allow clients to start watching at now.
	// NOTE: We assume that <store> is thread-safe.
	store store.Indexer

	// ResourceVersion of the last list result (populated via Replace() method).
	listResourceVersion uint64

	// Stores previous snapshots of orderedLister to allow serving requests from previous revisions.
	snapshots           store.Snapshotter
	snapshottingEnabled atomic.Bool
}

func (w *watchCacheStorage) SnapshotLocked() store.Snapshot {
	return w.store
}

func (w *watchCacheStorage) Compact(rev uint64) {
	if w.snapshots == nil {
		return
	}
	w.snapshots.RemoveLess(rev)
}

func (w *watchCacheStorage) MarkConsistent(consistent bool) {
	if utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
		w.snapshottingEnabled.Store(consistent)
		if !consistent && w.snapshots != nil {
			w.snapshots.Reset()
		}
	}
}

func (w *watchCacheStorage) getLatestSnapshotLocked(key, continueKey string) (store.Snapshot, error) {
	if w.snapshots != nil && w.snapshottingEnabled.Load() {
		snap, ok := w.snapshots.Latest()
		if ok {
			// Snapshots are added in order as we update store, so the
			// latest snapshot match latest store state and latest revision.
			return snap, nil
		}
	}
	// TODO: Consider using Indexer Clone() after benchmarking.
	snap, err := orderedSnapshotResponseFromIndexer(w.store, key, continueKey)
	if err != nil {
		return nil, err
	}
	return snap, nil
}

func orderedSnapshotResponseFromIndexer(indexer store.Indexer, key, continueKey string) (store.Snapshot, error) {
	items, err := indexer.OrderedListPrefix(key, continueKey)
	if err != nil {
		return nil, err
	}
	return orderedListSnapshot{Items: items}, nil
}

type orderedListSnapshot struct {
	Items []interface{}
}

var _ store.Snapshot = (*orderedListSnapshot)(nil)

func (o orderedListSnapshot) GetByKey(key string) (interface{}, bool, error) {
	for _, item := range o.Items {
		elem, ok := item.(*store.Element)
		if ok && elem.Key == key {
			return item, true, nil
		}
	}
	return nil, false, nil
}

func (o orderedListSnapshot) OrderedListPrefix(prefix, continueKey string) ([]interface{}, error) {
	return o.Items, nil
}

type listSnapshot struct {
	Items []interface{}
}

var _ store.Snapshot = (*listSnapshot)(nil)

func (l listSnapshot) GetByKey(key string) (interface{}, bool, error) {
	for _, item := range l.Items {
		elem, ok := item.(*store.Element)
		if ok && elem.Key == key {
			return item, true, nil
		}
	}
	return nil, false, nil
}

func (l listSnapshot) OrderedListPrefix(prefix string, continueKey string) ([]interface{}, error) {
	var result []interface{}
	for _, item := range l.Items {
		elem, ok := item.(*store.Element)
		if !ok {
			return nil, fmt.Errorf("non *store.Element returned from storage: %v", item)
		}
		if len(continueKey) > 0 && continueKey >= elem.Key {
			continue
		}
		if !key.HasPathPrefix(elem.Key, prefix) {
			continue
		}
		result = append(result, item)
	}
	sort.Sort(sortableStoreElements(result))
	return result, nil
}

type sortableStoreElements []interface{}

func (s sortableStoreElements) Len() int {
	return len(s)
}

func (s sortableStoreElements) Less(i, j int) bool {
	return s[i].(*store.Element).Key < s[j].(*store.Element).Key
}

func (s sortableStoreElements) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Get takes runtime.Object as a parameter. However, it returns
// pointer to <storeElement>.
func (w *watchCacheStorage) Get(obj interface{}) (interface{}, bool, error) {
	object, ok := obj.(runtime.Object)
	if !ok {
		return nil, false, fmt.Errorf("obj does not implement runtime.Object interface: %v", obj)
	}
	key, err := w.keyFunc(object)
	if err != nil {
		return nil, false, fmt.Errorf("couldn't compute key: %w", err)
	}

	return w.store.Get(&store.Element{Key: key, Object: object})
}

// GetByKey returns pointer to <storeElement>.
func (w *watchCacheStorage) GetByKey(key string) (interface{}, bool, error) {
	return w.store.GetByKey(key)
}

func (w *watchCacheStorage) ListKeys() []string {
	return w.store.ListKeys()
}

// List returns list of pointers to <store.Element> objects.
func (w *watchCacheStorage) List() []interface{} {
	return w.store.List()
}

// UpdateStoreLocked executes a mutation (Add, Update, Delete) on the underlying store.
func (w *watchCacheStorage) UpdateStoreLocked(eventType watch.EventType, elem *store.Element) error {
	switch eventType {
	case watch.Added:
		return w.store.Add(elem)
	case watch.Modified:
		return w.store.Update(elem)
	case watch.Deleted:
		return w.store.Delete(elem)
	default:
		return fmt.Errorf("unexpected event type: %v", eventType)
	}
}

// AddSnapshotLocked collects a new snapshot if snapshotting is enabled.
func (w *watchCacheStorage) AddSnapshotLocked(version uint64) {
	if w.snapshots != nil && w.snapshottingEnabled.Load() {
		w.snapshots.Add(version, w.store)
	}
}

// CompactSnapshotsLocked prunes snapshots older than the oldest history version.
func (w *watchCacheStorage) CompactSnapshotsLocked(oldestRV uint64) {
	if w.snapshots != nil && w.snapshottingEnabled.Load() {
		w.snapshots.RemoveLess(oldestRV)
	}
}

// ReplaceLocked replaces the elements in the underlying store and resets snapshots.
func (w *watchCacheStorage) ReplaceLocked(toReplace []interface{}, resourceVersion string, version uint64) error {
	if err := w.store.Replace(toReplace, resourceVersion); err != nil {
		return err
	}
	if w.snapshots != nil {
		w.snapshots.Reset()
		if w.snapshottingEnabled.Load() {
			w.snapshots.Add(version, w.store)
		}
	}
	w.listResourceVersion = version
	return nil
}

// GetExactSnapshotLocked retrieves a snapshot less than or equal to the given resource version.
func (w *watchCacheStorage) GetExactSnapshotLocked(resourceVersion uint64) (store.Snapshot, error) {
	if w.snapshots == nil {
		return nil, errors.NewResourceExpired(fmt.Sprintf("too old resource version: %d", resourceVersion))
	}
	snap, ok := w.snapshots.GetLessOrEqual(resourceVersion)
	if !ok {
		return nil, errors.NewResourceExpired(fmt.Sprintf("too old resource version: %d", resourceVersion))
	}
	return snap, nil
}

// ByIndex retrieves elements from the indexer by index name and value.
func (w *watchCacheStorage) ByIndex(indexName, value string) ([]interface{}, error) {
	return w.store.ByIndex(indexName, value)
}

// ListResourceVersion returns the list resource version.
func (w *watchCacheStorage) ListResourceVersion() uint64 {
	return w.listResourceVersion
}
