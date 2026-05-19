/*
Copyright 2025 The Kubernetes Authors.

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

package pullmanager

import (
	"fmt"
	"sync"
	"sync/atomic"

	"k8s.io/klog/v2"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/utils/lru"
)

type lruCache[K comparable, V any] struct {
	cache   *lru.Cache
	maxSize int

	// authoritative indicates if we can consider the cached records an
	// authoritative source.
	// False if the cache is evicted any records because it reached capacity, or
	// if there were errors during its initialization.
	authoritative atomic.Bool

	// deletingKeys is used by the eviction function to distinguish between keys
	// being explicitly deleted and keys being removed because the cache was too
	// large or being cleared.
	//
	// This is only modified by lruCache.Delete().
	deletingKeys sync.Map
}

func newLRUCache[K comparable, V any](size int) *lruCache[K, V] {
	c := lru.New(size)
	l := &lruCache[K, V]{
		maxSize:      size,
		cache:        c,
		deletingKeys: sync.Map{},
	}
	if err := c.SetEvictionFunc(func(key lru.Key, _ any) {
		if _, shouldIgnore := l.deletingKeys.Load(key); shouldIgnore {
			return
		}
		// any eviction makes our cache non-authoritative
		l.authoritative.Store(false)
	}); err != nil {
		panic(fmt.Sprintf("failed to set eviction function to the LRU cache: %v", err))
	}
	return l
}

func (c *lruCache[K, V]) Get(key K) (*V, bool) {
	value, found := c.cache.Get(key)
	if !found {
		return nil, false
	}
	if value == nil {
		return nil, true
	}
	return value.(*V), true
}

func (c *lruCache[K, V]) Set(key K, value *V) { c.cache.Add(key, value) }
func (c *lruCache[K, V]) Len() int            { return c.cache.Len() }
func (c *lruCache[K, V]) Clear()              { c.cache.Clear() }

// Delete will prevent authoritative cache status changes.
//
// Must be called locked with an external write lock on `key`.
func (c *lruCache[K, V]) Delete(key K) {
	c.deletingKeys.Store(key, struct{}{})
	defer c.deletingKeys.Delete(key)
	c.cache.Remove(key)
}

// cachedPullRecordsAccessor implements a write-through cache layer on top
// of another PullRecordsAccessor
type cachedPullRecordsAccessor struct {
	delegate PullRecordsAccessor

	intentsLocks       *StripedLockSet
	intents            *lruCache[string, kubeletconfiginternal.ImagePullIntent]
	pulledRecordsLocks *StripedLockSet
	pulledRecords      *lruCache[string, kubeletconfiginternal.ImagePulledRecord]
}

func NewCachedPullRecordsAccessor(delegate PullRecordsAccessor, intentsCacheSize, pulledRecordsCacheSize, stripedLocksSize int32) PullRecordsAccessor {
	intentsCacheSize = min(intentsCacheSize, 1024)
	pulledRecordsCacheSize = min(pulledRecordsCacheSize, 2000)

	c := &cachedPullRecordsAccessor{
		delegate: delegate,

		intentsLocks:       NewStripedLockSet(stripedLocksSize),
		intents:            newLRUCache[string, kubeletconfiginternal.ImagePullIntent](int(intentsCacheSize)),
		pulledRecordsLocks: NewStripedLockSet(stripedLocksSize),
		pulledRecords:      newLRUCache[string, kubeletconfiginternal.ImagePulledRecord](int(pulledRecordsCacheSize)),
	}
	// warm our caches and set authoritative
	_, err := c.ListImagePullIntents()
	if err != nil {
		klog.InfoS("there was an error initializing the image pull intents cache, the cache will work in a non-authoritative mode until the intents are listed successfully", "error", err)
	}
	_, err = c.ListImagePulledRecords()
	if err != nil {
		klog.InfoS("there was an error initializing the image pulled records cache, the cache will work in a non-authoritative mode until the pulled records are listed successfully", "error", err)
	}
	return NewMeteringRecordsAccessor(c, inMemIntentsPercent, inMemPulledRecordsPercent)
}

func (c *cachedPullRecordsAccessor) ListImagePullIntents() ([]*kubeletconfiginternal.ImagePullIntent, error) {
	return cacheRefreshingList(
		c.intents,
		c.intentsLocks,
		c.delegate.ListImagePullIntents,
		pullIntentToCacheKey,
	)
}

func (c *cachedPullRecordsAccessor) ImagePullIntentExists(image string) (bool, error) {
	// do the cheap Get() lock-free
	if _, exists := c.intents.Get(image); exists {
		return true, nil
	}

	// on a miss, lock on the image
	c.intentsLocks.Lock(image)
	defer c.intentsLocks.Unlock(image)

	// check again if the image exists in the cache under image lock
	if _, exists := c.intents.Get(image); exists {
		return true, nil
	}
	// if the cache is authoritative, return false on a miss
	if c.intents.authoritative.Load() {
		return false, nil
	}

	// fall through to the expensive lookup
	exists, err := c.delegate.ImagePullIntentExists(image)
	if err == nil && exists {
		c.intents.Set(image, &kubeletconfiginternal.ImagePullIntent{
			Image: image,
		})
	}
	return exists, err
}

func (c *cachedPullRecordsAccessor) WriteImagePullIntent(image string) error {
	c.intentsLocks.Lock(image)
	defer c.intentsLocks.Unlock(image)

	if err := c.delegate.WriteImagePullIntent(image); err != nil {
		return err
	}
	c.intents.Set(image, &kubeletconfiginternal.ImagePullIntent{
		Image: image,
	})

	return nil
}

func (c *cachedPullRecordsAccessor) DeleteImagePullIntent(image string) error {
	c.intentsLocks.Lock(image)
	defer c.intentsLocks.Unlock(image)

	if err := c.delegate.DeleteImagePullIntent(image); err != nil {
		return err
	}
	c.intents.Delete(image)
	return nil
}

func (c *cachedPullRecordsAccessor) ListImagePulledRecords() ([]*kubeletconfiginternal.ImagePulledRecord, error) {
	return cacheRefreshingList(
		c.pulledRecords,
		c.pulledRecordsLocks,
		c.delegate.ListImagePulledRecords,
		pulledRecordToCacheKey,
	)
}

func (c *cachedPullRecordsAccessor) GetImagePulledRecord(imageRef string) (*kubeletconfiginternal.ImagePulledRecord, bool, error) {
	// do the cheap Get() lock-free
	pulledRecord, exists := c.pulledRecords.Get(imageRef)
	if exists {
		return pulledRecord, true, nil
	}

	// on a miss, lock on the imageRef
	c.pulledRecordsLocks.Lock(imageRef)
	defer c.pulledRecordsLocks.Unlock(imageRef)

	// check again if the imageRef exists in the cache under imageRef lock
	pulledRecord, exists = c.pulledRecords.Get(imageRef)
	if exists {
		return pulledRecord, true, nil
	}
	// if the cache is authoritative, return false on a miss
	if c.pulledRecords.authoritative.Load() {
		return nil, false, nil
	}

	// fall through to the expensive lookup
	pulledRecord, exists, err := c.delegate.GetImagePulledRecord(imageRef)
	if err == nil && exists {
		c.pulledRecords.Set(imageRef, pulledRecord)
	}
	return pulledRecord, exists, err
}

func (c *cachedPullRecordsAccessor) WriteImagePulledRecord(record *kubeletconfiginternal.ImagePulledRecord) error {
	c.pulledRecordsLocks.Lock(record.ImageRef)
	defer c.pulledRecordsLocks.Unlock(record.ImageRef)

	if err := c.delegate.WriteImagePulledRecord(record); err != nil {
		return err
	}
	c.pulledRecords.Set(record.ImageRef, record)
	return nil
}

func (c *cachedPullRecordsAccessor) DeleteImagePulledRecord(imageRef string) error {
	c.pulledRecordsLocks.Lock(imageRef)
	defer c.pulledRecordsLocks.Unlock(imageRef)

	if err := c.delegate.DeleteImagePulledRecord(imageRef); err != nil {
		return err
	}
	c.pulledRecords.Delete(imageRef)
	return nil
}

func (f *cachedPullRecordsAccessor) intentsSize() (uint, error) {
	intentsUsage := f.intents.Len() * 100 / f.intents.maxSize
	return uint(intentsUsage), nil
}

func (f *cachedPullRecordsAccessor) pulledRecordsSize() (uint, error) {
	pulledRecordsUsage := f.pulledRecords.Len() * 100 / f.pulledRecords.maxSize
	return uint(pulledRecordsUsage), nil
}

func cacheRefreshingList[K comparable, V any](
	cache *lruCache[K, V],
	delegateLocks *StripedLockSet,
	listRecordsFunc func() ([]*V, error),
	recordToKey func(*V) K,
) ([]*V, error) {
	wasAuthoritative := cache.authoritative.Load()
	if !wasAuthoritative {
		// doing a full list gives us an opportunity to become authoritative
		// if we get back an error-free result that fits in our cache
		delegateLocks.GlobalLock()
		defer delegateLocks.GlobalUnlock()
	}

	results, err := listRecordsFunc()
	if wasAuthoritative {
		return results, err
	}

	resultsAreAuthoritative := err == nil && len(results) < cache.maxSize
	// populate the cache if that would make our cache authoritative or if the cache is currently empty
	if resultsAreAuthoritative || cache.Len() == 0 {
		cache.Clear()
		// populate up to maxSize results in the cache
		for _, record := range results[:min(len(results), cache.maxSize)] {
			cache.Set(recordToKey(record), record)
		}
		cache.authoritative.Store(resultsAreAuthoritative)
	}

	return results, err
}

func pullIntentToCacheKey(intent *kubeletconfiginternal.ImagePullIntent) string {
	return intent.Image
}

func pulledRecordToCacheKey(record *kubeletconfiginternal.ImagePulledRecord) string {
	return record.ImageRef
}
