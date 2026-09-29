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
	"sort"
	"sync"
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
		store:               newBtreeStore(btreeDegree),
		indexer:             newIndexer(ElementIndexers(indexers)),
		snapshots:           newSnapshotter(),
		listResourceVersion: 0,
	}
	storage.latestSnapshot.Store(storage.store.Clone())
	if utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
		storage.snapshottingEnabled = true
	}
	return storage
}

type WatchCacheStorage struct {
	keyFunc func(runtime.Object) (string, error)

	// ResourceVersion of the last list result (populated via ReplaceLocked() method).
	listResourceVersion uint64

	// latestSnapshot is an immutable clone of store, republished under lock
	// after every write, so reads of the latest state don't need lock.
	latestSnapshot atomic.Pointer[btreeStore]

	// Access to store, indexer, and snapshots is synchronized using lock.
	lock                sync.RWMutex
	store               btreeStore
	indexer             indexer
	snapshottingEnabled bool
	snapshots           snapshotter
}

func (w *WatchCacheStorage) SnapshottingEnabled() bool {
	w.lock.RLock()
	defer w.lock.RUnlock()
	return w.snapshottingEnabled
}

func (w *WatchCacheStorage) CanServeExactRV(rv uint64) bool {
	w.lock.RLock()
	defer w.lock.RUnlock()
	if !w.snapshottingEnabled {
		return false
	}
	_, canServe := w.snapshots.GetLessOrEqual(rv)
	return canServe
}

func (w *WatchCacheStorage) UpdateListResourceVersion(rv uint64) {
	w.listResourceVersion = rv
}

func (w *WatchCacheStorage) Compact(rev uint64) {
	w.lock.Lock()
	defer w.lock.Unlock()
	if !w.snapshottingEnabled {
		return
	}
	w.snapshots.RemoveLess(rev)
}

func (w *WatchCacheStorage) MarkConsistent(consistent bool) {
	if utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
		w.lock.Lock()
		defer w.lock.Unlock()
		w.snapshottingEnabled = consistent
		if !consistent {
			w.snapshots.Reset()
		}
	}
}

func (w *WatchCacheStorage) LatestSnapshot() Snapshot {
	return w.latestSnapshot.Load()
}

// listSnapshot serves an unordered index bucket.
type listSnapshot struct {
	Items           []interface{}
	resourceVersion uint64
}

var _ Snapshot = (*listSnapshot)(nil)

func (l listSnapshot) ResourceVersion() uint64 {
	return l.resourceVersion
}

func (l listSnapshot) GetByKey(key string) (interface{}, bool, error) {
	for _, item := range l.Items {
		elem, ok := item.(*Element)
		if ok && elem.Key == key {
			return item, true, nil
		}
	}
	return nil, false, nil
}

func (l listSnapshot) OrderedListPrefix(prefix string, continueKey string) ([]interface{}, error) {
	var result []interface{}
	for _, item := range l.Items {
		elem, ok := item.(*Element)
		if !ok {
			return nil, fmt.Errorf("non *Element returned from storage: %v", item)
		}
		if len(continueKey) > 0 && continueKey > elem.Key {
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

func (l listSnapshot) RangePrefix(prefix, continueKey string) Range {
	items, err := l.OrderedListPrefix(prefix, continueKey)
	if err != nil {
		return failedRange{err}
	}
	elems := make(elements, 0, len(items))
	for _, item := range items {
		// OrderedListPrefix has already checked every item is an *Element.
		elems = append(elems, item.(*Element))
	}
	return elems
}

type failedRange struct{ err error }

func (r failedRange) All() iter.Seq2[*Element, error] {
	return func(yield func(*Element, error) bool) { yield(nil, r.err) }
}

func (r failedRange) Count() int {
	return 0
}

type sortableStoreElements []interface{}

func (s sortableStoreElements) Len() int {
	return len(s)
}

func (s sortableStoreElements) Less(i, j int) bool {
	return s[i].(*Element).Key < s[j].(*Element).Key
}

func (s sortableStoreElements) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
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

	return w.get(&Element{Key: key, Object: object})
}

func (w *WatchCacheStorage) get(obj interface{}) (item interface{}, exists bool, err error) {
	return w.latestSnapshot.Load().Get(obj)
}

// GetByKey returns pointer to <storeElement>.
func (w *WatchCacheStorage) GetByKey(key string) (item interface{}, exists bool, err error) {
	return w.latestSnapshot.Load().GetByKey(key)
}

func (w *WatchCacheStorage) OrderedListPrefix(prefix, continueKey string) ([]interface{}, error) {
	return w.latestSnapshot.Load().OrderedListPrefix(prefix, continueKey)
}

func (w *WatchCacheStorage) ListKeys() []string {
	return w.latestSnapshot.Load().ListKeys()
}

// List returns list of pointers to <Element> objects.
func (w *WatchCacheStorage) List() []interface{} {
	return w.latestSnapshot.Load().List()
}

// UpdateStore executes a mutation (Add, Update, Delete) on the underlying store.
// It returns the element that was previously stored under the same key, if any.
func (w *WatchCacheStorage) UpdateStore(eventType watch.EventType, elem *Element, resourceVersion uint64) (prev *Element, err error) {
	if elem == nil {
		return nil, fmt.Errorf("elem cannot be nil")
	}
	w.lock.Lock()
	defer w.lock.Unlock()
	switch eventType {
	case watch.Added, watch.Modified:
		prev = w.store.addOrUpdateElem(elem, resourceVersion)
		err = w.indexer.updateElem(elem.Key, prev, elem, resourceVersion)
	case watch.Deleted:
		prev = w.store.deleteElem(elem, resourceVersion)
		err = w.indexer.updateElem(elem.Key, prev, nil, resourceVersion)
	default:
		err = fmt.Errorf("unexpected event type: %v", eventType)
	}
	if err != nil {
		return nil, err
	}
	latest := w.store.Clone()
	w.latestSnapshot.Store(latest)
	if w.snapshottingEnabled {
		w.snapshots.Add(resourceVersion, latest)
	}
	return prev, nil
}

func (w *WatchCacheStorage) UpdateResourceVersion(resourceVersion uint64) {
	w.lock.Lock()
	defer w.lock.Unlock()
	w.store.resourceVersion = resourceVersion
	w.indexer.resourceVersion = resourceVersion
	w.latestSnapshot.Store(w.store.Clone())
}

// CompactSnapshotsLocked prunes snapshots older than the oldest history version.
func (w *WatchCacheStorage) CompactSnapshotsLocked(oldestRV uint64) {
	w.Compact(oldestRV)
}

// Replace replaces the elements in the underlying store and resets snapshots.
func (w *WatchCacheStorage) Replace(toReplace []*Element, version uint64) error {
	w.lock.Lock()
	defer w.lock.Unlock()
	w.store.Replace(toReplace, version)
	if err := w.indexer.Replace(toReplace, version); err != nil {
		return err
	}
	w.snapshots.Reset()
	latest := w.store.Clone()
	w.latestSnapshot.Store(latest)
	if w.snapshottingEnabled {
		w.snapshots.Add(version, latest)
	}
	w.listResourceVersion = version
	return nil
}

// GetExactSnapshotLocked retrieves a snapshot less than or equal to the given resource version.
func (w *WatchCacheStorage) GetExactSnapshotLocked(resourceVersion uint64) (Snapshot, error) {
	w.lock.RLock()
	defer w.lock.RUnlock()
	if !w.snapshottingEnabled {
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
	w.lock.RLock()
	defer w.lock.RUnlock()
	result, resourceVersion, err := w.indexer.ByIndex(indexName, value)
	if err != nil {
		return nil, err
	}
	return listSnapshot{Items: result, resourceVersion: resourceVersion}, nil
}

// ListResourceVersion returns the list resource version.
func (w *WatchCacheStorage) ListResourceVersion() uint64 {
	return w.listResourceVersion
}
