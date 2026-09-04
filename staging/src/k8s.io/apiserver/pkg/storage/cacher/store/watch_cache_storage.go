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
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/cacher/key"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
)

func NewWatchCacheStorage(keyFunc func(runtime.Object) (string, error), indexers *cache.Indexers) *WatchCacheStorage {
	storage := &WatchCacheStorage{
		keyFunc:             keyFunc,
		store:               NewIndexer(indexers),
		listResourceVersion: 0,
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
		storage.snapshottingEnabled.Store(true)
		storage.snapshots = NewSnapshotter()
	}
	return storage
}

type WatchCacheStorage struct {
	keyFunc func(runtime.Object) (string, error)

	// store will effectively support LIST operation from the "end of cache
	// history" i.e. from the moment just after the newest cached watched event.
	// It is necessary to effectively allow clients to start watching at now.
	// NOTE: We assume that <store> is thread-safe.
	store Indexer

	// ResourceVersion of the last list result (populated via Replace() method).
	listResourceVersion uint64

	// Stores previous snapshots of orderedLister to allow serving requests from previous revisions.
	snapshots           Snapshotter
	snapshottingEnabled atomic.Bool
}

// StoreLocked returns the live store.
// Unlike GetExactSnapshotLocked this is not an immutable point-in-time copy.
// The caller must hold the lock for the duration of use.
func (w *WatchCacheStorage) StoreLocked() Indexer {
	return w.store
}

func (w *WatchCacheStorage) SnapshottingEnabled() bool {
	return w.snapshots != nil && w.snapshottingEnabled.Load()
}

func (w *WatchCacheStorage) CanServeExactRV(rv uint64) bool {
	if w.snapshots == nil {
		return false
	}
	_, canServe := w.snapshots.GetLessOrEqual(rv)
	return canServe
}

func (w *WatchCacheStorage) UpdateListResourceVersion(rv uint64) {
	w.listResourceVersion = rv
}

func (w *WatchCacheStorage) Compact(rev uint64) {
	if w.snapshots == nil {
		return
	}
	w.snapshots.RemoveLess(rev)
}

func (w *WatchCacheStorage) MarkConsistent(consistent bool) {
	if utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
		w.snapshottingEnabled.Store(consistent)
		if !consistent && w.snapshots != nil {
			w.snapshots.Reset()
		}
	}
}

func (w *WatchCacheStorage) LatestSnapshotLocked() (Snapshot, bool) {
	if w.SnapshottingEnabled() {
		return w.snapshots.Latest()
	}
	return nil, false
}

func (w *WatchCacheStorage) GetLatestSnapshotOrBuildLocked(key, continueKey string) (Snapshot, error) {
	if snap, ok := w.LatestSnapshotLocked(); ok {
		// Snapshots are added in order as we update store, so the
		// latest snapshot match latest store state and latest revision.
		return snap, nil
	}
	// TODO: Consider using Indexer Clone() after benchmarking.
	return orderedSnapshotResponseFromIndexer(w.store, key, continueKey)
}

func orderedSnapshotResponseFromIndexer(indexer Indexer, key, continueKey string) (Snapshot, error) {
	items, err := indexer.OrderedListPrefix(key, continueKey)
	if err != nil {
		return nil, err
	}
	elems := make(orderedListSnapshot, 0, len(items))
	for _, item := range items {
		elem, ok := item.(*Element)
		if !ok {
			return nil, fmt.Errorf("non *Element returned from storage: %v", item)
		}
		elems = append(elems, elem)
	}
	return elems, nil
}

// orderedListSnapshot serves a key-ordered slice copied out of the store.
type orderedListSnapshot []*Element

func (o orderedListSnapshot) GetByKey(key string) (interface{}, bool, error) {
	if i, found := slices.BinarySearchFunc(o, key, compareKey); found {
		return o[i], true, nil
	}
	return nil, false, nil
}

func (o orderedListSnapshot) OrderedListPrefix(prefix, continueKey string) ([]interface{}, error) {
	return listOf(o.RangePrefix(prefix, continueKey)), nil
}

func (o orderedListSnapshot) RangePrefix(prefix, continueKey string) Range {
	start, _ := slices.BinarySearchFunc(o, max(prefix, continueKey), compareKey)
	end := start
	for end < len(o) && strings.HasPrefix(o[end].Key, prefix) {
		end++
	}
	return orderedElements(o[start:end])
}

func compareKey(elem *Element, key string) int {
	return strings.Compare(elem.Key, key)
}

// listSnapshot serves an unordered index bucket.
type listSnapshot []*Element

func (l listSnapshot) GetByKey(key string) (interface{}, bool, error) {
	for _, elem := range l {
		if elem.Key == key {
			return elem, true, nil
		}
	}
	return nil, false, nil
}

func (l listSnapshot) OrderedListPrefix(prefix string, continueKey string) ([]interface{}, error) {
	return listOf(l.RangePrefix(prefix, continueKey)), nil
}

func (l listSnapshot) RangePrefix(prefix, continueKey string) Range {
	var elems orderedElements
	for _, elem := range l {
		if continueKey <= elem.Key && key.HasPathPrefix(elem.Key, prefix) {
			elems = append(elems, elem)
		}
	}
	slices.SortFunc(elems, func(a, b *Element) int { return strings.Compare(a.Key, b.Key) })
	return elems
}

type orderedElements []*Element

func (o orderedElements) All() iter.Seq[*Element] {
	return slices.Values(o)
}

func (o orderedElements) Count() int {
	return len(o)
}

func listOf(r Range) []interface{} {
	items := make([]interface{}, 0, r.Count())
	for elem := range r.All() {
		items = append(items, elem)
	}
	return items
}

// Get takes runtime.Object as a parameter. However, it returns
// pointer to <storeElement>.
func (w *WatchCacheStorage) Get(obj interface{}) (interface{}, bool, error) {
	object, ok := obj.(runtime.Object)
	if !ok {
		return nil, false, fmt.Errorf("obj does not implement runtime.Object interface: %v", obj)
	}
	key, err := w.keyFunc(object)
	if err != nil {
		return nil, false, fmt.Errorf("couldn't compute key: %w", err)
	}

	return w.store.Get(&Element{Key: key, Object: object})
}

// GetByKey returns pointer to <storeElement>.
func (w *WatchCacheStorage) GetByKey(key string) (interface{}, bool, error) {
	return w.store.GetByKey(key)
}

func (w *WatchCacheStorage) ListKeys() []string {
	return w.store.ListKeys()
}

// List returns list of pointers to <Element> objects.
func (w *WatchCacheStorage) List() []interface{} {
	return w.store.List()
}

// UpdateStoreLocked executes a mutation (Add, Update, Delete) on the underlying store.
func (w *WatchCacheStorage) UpdateStoreLocked(eventType watch.EventType, elem *Element, resourceVersion uint64) (err error) {
	switch eventType {
	case watch.Added:
		err = w.store.Add(elem)
	case watch.Modified:
		err = w.store.Update(elem)
	case watch.Deleted:
		err = w.store.Delete(elem)
	default:
		err = fmt.Errorf("unexpected event type: %v", eventType)
	}
	if err != nil {
		return err
	}
	if w.snapshots != nil && w.snapshottingEnabled.Load() {
		w.snapshots.Add(resourceVersion, w.store)
	}
	return nil
}

// CompactSnapshotsLocked prunes snapshots older than the oldest history version.
func (w *WatchCacheStorage) CompactSnapshotsLocked(oldestRV uint64) {
	if w.snapshots != nil && w.snapshottingEnabled.Load() {
		w.snapshots.RemoveLess(oldestRV)
	}
}

// ReplaceLocked replaces the elements in the underlying store and resets snapshots.
func (w *WatchCacheStorage) ReplaceLocked(toReplace []interface{}, resourceVersion string, version uint64) error {
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
func (w *WatchCacheStorage) GetExactSnapshotLocked(resourceVersion uint64) (Snapshot, error) {
	if w.snapshots == nil {
		return nil, errors.NewResourceExpired(fmt.Sprintf("too old resource version: %d", resourceVersion))
	}
	snap, ok := w.snapshots.GetLessOrEqual(resourceVersion)
	if !ok {
		return nil, errors.NewResourceExpired(fmt.Sprintf("too old resource version: %d", resourceVersion))
	}
	return snap, nil
}

// GetByIndexSnapshot retrieves elements by index and wraps them in a Snapshot.
func (w *WatchCacheStorage) GetByIndexSnapshot(indexName, value string) (Snapshot, error) {
	result, err := w.store.ByIndex(indexName, value)
	if err != nil {
		return nil, err
	}
	return listSnapshot(result), nil
}

// ListResourceVersion returns the list resource version.
func (w *WatchCacheStorage) ListResourceVersion() uint64 {
	return w.listResourceVersion
}
