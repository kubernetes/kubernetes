// An LRU cached aimed at high concurrency
package ccache

import (
	"container/list"
	"hash/fnv"
	"sync/atomic"
	"time"
)

type LayeredCache struct {
	*Configuration
	list        *list.List
	buckets     []*layeredBucket
	bucketMask  uint32
	size        int64
	deletables  chan *Item
	promotables chan *Item
	donec       chan struct{}
}

// Create a new layered cache with the specified configuration.
// A layered cache used a two keys to identify a value: a primary key
// and a secondary key. Get, Set and Delete require both a primary and
// secondary key. However, DeleteAll requires only a primary key, deleting
// all values that share the same primary key.

// Layered Cache is useful as an HTTP cache, where an HTTP purge might
// delete multiple variants of the same resource:
// primary key = "user/44"
// secondary key 1 = ".json"
// secondary key 2 = ".xml"

// See ccache.Configure() for creating a configuration
func Layered(config *Configuration) *LayeredCache {
	c := &LayeredCache{
		list:          list.New(),
		Configuration: config,
		bucketMask:    uint32(config.buckets) - 1,
		buckets:       make([]*layeredBucket, config.buckets),
		deletables:    make(chan *Item, config.deleteBuffer),
	}
	for i := 0; i < int(config.buckets); i++ {
		c.buckets[i] = &layeredBucket{
			buckets: make(map[string]*bucket),
		}
	}
	c.restart()
	return c
}

// Get an item from the cache. Returns nil if the item wasn't found.
// This can return an expired item. Use item.Expired() to see if the item
// is expired and item.TTL() to see how long until the item expires (which
// will be negative for an already expired item).
func (c *LayeredCache) Get(primary, secondary string) *Item {
	item := c.bucket(primary).get(primary, secondary)
	if item == nil {
		return nil
	}
	if item.expires > time.Now().UnixNano() {
		c.promote(item)
	}
	return item
}

// Get the secondary cache for a given primary key. This operation will
// never return nil. In the case where the primary key does not exist, a
// new, underlying, empty bucket will be created and returned.
func (c *LayeredCache) GetOrCreateSecondaryCache(primary string) *SecondaryCache {
	primaryBkt := c.bucket(primary)
	bkt := primaryBkt.getSecondaryBucket(primary)
	primaryBkt.Lock()
	if bkt == nil {
		bkt = &bucket{lookup: make(map[string]*Item)}
		primaryBkt.buckets[primary] = bkt
	}
	primaryBkt.Unlock()
	return &SecondaryCache{
		bucket: bkt,
		pCache: c,
	}
}

// Used when the cache was created with the Track() configuration option.
// Avoid otherwise
func (c *LayeredCache) TrackingGet(primary, secondary string) TrackedItem {
	item := c.Get(primary, secondary)
	if item == nil {
		return NilTracked
	}
	item.track()
	return item
}

// Set the value in the cache for the specified duration
func (c *LayeredCache) Set(primary, secondary string, value interface{}, duration time.Duration) {
	c.set(primary, secondary, value, duration)
}

// Replace the value if it exists, does not set if it doesn't.
// Returns true if the item existed an was replaced, false otherwise.
// Replace does not reset item's TTL nor does it alter its position in the LRU
func (c *LayeredCache) Replace(primary, secondary string, value interface{}) bool {
	item := c.bucket(primary).get(primary, secondary)
	if item == nil {
		return false
	}
	c.Set(primary, secondary, value, item.TTL())
	return true
}

// Attempts to get the value from the cache and calles fetch on a miss.
// If fetch returns an error, no value is cached and the error is returned back
// to the caller.
func (c *LayeredCache) Fetch(primary, secondary string, duration time.Duration, fetch func() (interface{}, error)) (*Item, error) {
	item := c.Get(primary, secondary)
	if item != nil {
		return item, nil
	}
	value, err := fetch()
	if err != nil {
		return nil, err
	}
	return c.set(primary, secondary, value, duration), nil
}

// Remove the item from the cache, return true if the item was present, false otherwise.
func (c *LayeredCache) Delete(primary, secondary string) bool {
	item := c.bucket(primary).delete(primary, secondary)
	if item != nil {
		c.deletables <- item
		return true
	}
	return false
}

// Deletes all items that share the same primary key
func (c *LayeredCache) DeleteAll(primary string) bool {
	return c.bucket(primary).deleteAll(primary, c.deletables)
}

//this isn't thread safe. It's meant to be called from non-concurrent tests
func (c *LayeredCache) Clear() {
	for _, bucket := range c.buckets {
		bucket.clear()
	}
	c.size = 0
	c.list = list.New()
}

func (c *LayeredCache) Stop() {
	close(c.promotables)
	<-c.donec
}

func (c *LayeredCache) restart() {
	c.promotables = make(chan *Item, c.promoteBuffer)
	c.donec = make(chan struct{})
	go c.worker()
}

func (c *LayeredCache) set(primary, secondary string, value interface{}, duration time.Duration) *Item {
	item, existing := c.bucket(primary).set(primary, secondary, value, duration)
	if existing != nil {
		c.deletables <- existing
	}
	c.promote(item)
	return item
}

func (c *LayeredCache) bucket(key string) *layeredBucket {
	h := fnv.New32a()
	h.Write([]byte(key))
	return c.buckets[h.Sum32()&c.bucketMask]
}

func (c *LayeredCache) promote(item *Item) {
	c.promotables <- item
}

func (c *LayeredCache) worker() {
	defer close(c.donec)
	for {
		select {
		case item, ok := <-c.promotables:
			if ok == false {
				return
			}
			if c.doPromote(item) && c.size > c.maxSize {
				c.gc()
			}
		case item := <-c.deletables:
			if item.element == nil {
				item.promotions = -2
			} else {
				c.size -= item.size
				c.list.Remove(item.element)
			}
		}
	}
}

func (c *LayeredCache) doPromote(item *Item) bool {
	// deleted before it ever got promoted
	if item.promotions == -2 {
		return false
	}
	if item.element != nil { //not a new item
		if item.shouldPromote(c.getsPerPromote) {
			c.list.MoveToFront(item.element)
			item.promotions = 0
		}
		return false
	}
	c.size += item.size
	item.element = c.list.PushFront(item)
	return true
}

func (c *LayeredCache) gc() {
	element := c.list.Back()
	for i := 0; i < c.itemsToPrune; i++ {
		if element == nil {
			return
		}
		prev := element.Prev()
		item := element.Value.(*Item)
		if c.tracking == false || atomic.LoadInt32(&item.refCount) == 0 {
			c.bucket(item.group).delete(item.group, item.key)
			c.size -= item.size
			c.list.Remove(element)
			item.promotions = -2
		}
		element = prev
	}
}
