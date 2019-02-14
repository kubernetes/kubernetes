/*
Copyright 2014 The Kubernetes Authors.

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

package cache

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/klog"
)

// ExpirationCache implements the store interface
//	1. All entries are automatically time stamped on insert
//		a. The key is computed based off the original item/keyFunc
//		b. The value inserted under that key is the timestamped item
//	2. Expiration happens lazily on read based on the expiration policy
//      a. No item can be inserted into the store while we're expiring
//		   *any* item in the cache.
//	3. Time-stamps are stripped off unexpired entries before return
// Note that the ExpirationCache is inherently slower than a normal
// threadSafeStore because it takes a write lock every time it checks if
// an item has expired.
type ExpirationCache struct {
	cacheStorage     ThreadSafeStore
	keyFunc          KeyFunc
	clock            clock.Clock
	expirationPolicy ExpirationPolicy
	// expirationLock is a write lock used to guarantee that we don't clobber
	// newly inserted objects because of a stale expiration timestamp comparison
	expirationLock sync.Mutex
}

// ExpirationPolicy dictates when an object expires. Currently only abstracted out
// so unittests don't rely on the system clock.
type ExpirationPolicy interface {
	IsExpired(obj *timestampedEntry) bool
}

// TTLPolicy implements a ttl based ExpirationPolicy.
type TTLPolicy struct {
	//	 >0: Expire entries with an age > ttl
	//	<=0: Don't expire any entry
	Ttl time.Duration

	// Clock used to calculate ttl expiration
	Clock clock.Clock
}

// IsExpired returns true if the given object is older than the ttl, or it can't
// determine its age.
func (p *TTLPolicy) IsExpired(obj *timestampedEntry) bool {
	return p.Ttl > 0 && p.Clock.Since(obj.timestamp) > p.Ttl
}

// timestampedEntry is the only type allowed in a ExpirationCache.
type timestampedEntry struct {
	obj       interface{}
	timestamp time.Time
}

// getTimestampedEntry returns the timestampedEntry stored under the given key.
func (c *ExpirationCache) getTimestampedEntry(key string) (*timestampedEntry, bool) {
	item, _ := c.cacheStorage.Get(key)
	if tsEntry, ok := item.(*timestampedEntry); ok {
		return tsEntry, true
	}
	return nil, false
}

// getOrExpire retrieves the object from the timestampedEntry if and only if it hasn't
// already expired. It holds a write lock across deletion.
func (c *ExpirationCache) getOrExpire(key string) (interface{}, bool) {
	// Prevent all inserts from the time we deem an item as "expired" to when we
	// delete it, so an un-expired item doesn't sneak in under the same key, just
	// before the Delete.
	c.expirationLock.Lock()
	defer c.expirationLock.Unlock()
	timestampedItem, exists := c.getTimestampedEntry(key)
	if !exists {
		return nil, false
	}
	if c.expirationPolicy.IsExpired(timestampedItem) {
		klog.V(4).Infof("Entry %v: %+v has expired", key, timestampedItem.obj)
		c.cacheStorage.Delete(key)
		return nil, false
	}
	return timestampedItem.obj, true
}

// GetByKey returns the item stored under the key, or sets exists=false.
func (c *ExpirationCache) GetByKey(key string) (interface{}, bool, error) {
	obj, exists := c.getOrExpire(key)
	return obj, exists, nil
}

// Get returns unexpired items. It purges the cache of expired items in the
// process.
func (c *ExpirationCache) Get(obj interface{}) (interface{}, bool, error) {
	key, err := c.keyFunc(obj)
	if err != nil {
		return nil, false, KeyError{obj, err}
	}
	obj, exists := c.getOrExpire(key)
	return obj, exists, nil
}

// List retrieves a list of unexpired items. It purges the cache of expired
// items in the process.
func (c *ExpirationCache) List() []interface{} {
	items := c.cacheStorage.List()

	list := make([]interface{}, 0, len(items))
	for _, item := range items {
		obj := item.(*timestampedEntry).obj
		if key, err := c.keyFunc(obj); err != nil {
			list = append(list, obj)
		} else if obj, exists := c.getOrExpire(key); exists {
			list = append(list, obj)
		}
	}
	return list
}

// ListKeys returns a list of all keys in the expiration cache.
func (c *ExpirationCache) ListKeys() []string {
	return c.cacheStorage.ListKeys()
}

// Add timestamps an item and inserts it into the cache, overwriting entries
// that might exist under the same key.
func (c *ExpirationCache) Add(obj interface{}) error {
	key, err := c.keyFunc(obj)
	if err != nil {
		return KeyError{obj, err}
	}
	c.expirationLock.Lock()
	defer c.expirationLock.Unlock()

	c.cacheStorage.Add(key, &timestampedEntry{obj, c.clock.Now()})
	return nil
}

// Update has not been implemented yet for lack of a use case, so this method
// simply calls `Add`. This effectively refreshes the timestamp.
func (c *ExpirationCache) Update(obj interface{}) error {
	return c.Add(obj)
}

// Delete removes an item from the cache.
func (c *ExpirationCache) Delete(obj interface{}) error {
	key, err := c.keyFunc(obj)
	if err != nil {
		return KeyError{obj, err}
	}
	c.expirationLock.Lock()
	defer c.expirationLock.Unlock()
	c.cacheStorage.Delete(key)
	return nil
}

// Replace will convert all items in the given list to TimestampedEntries
// before attempting the replace operation. The replace operation will
// delete the contents of the ExpirationCache `c`.
func (c *ExpirationCache) Replace(list []interface{}, resourceVersion string) error {
	items := make(map[string]interface{}, len(list))
	ts := c.clock.Now()
	for _, item := range list {
		key, err := c.keyFunc(item)
		if err != nil {
			return KeyError{item, err}
		}
		items[key] = &timestampedEntry{item, ts}
	}
	c.expirationLock.Lock()
	defer c.expirationLock.Unlock()
	c.cacheStorage.Replace(items, resourceVersion)
	return nil
}

// Resync will touch all objects to put them into the processing queue
func (c *ExpirationCache) Resync() error {
	return c.cacheStorage.Resync()
}

// NewTTLStore creates and returns a ExpirationCache with a TTLPolicy
func NewTTLStore(keyFunc KeyFunc, ttl time.Duration) Store {
	return &ExpirationCache{
		cacheStorage:     NewThreadSafeStore(Indexers{}, Indices{}),
		keyFunc:          keyFunc,
		clock:            clock.RealClock{},
		expirationPolicy: &TTLPolicy{ttl, clock.RealClock{}},
	}
}
