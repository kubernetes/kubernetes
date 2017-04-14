// An LRU cached aimed at high concurrency
package ccache

import (
	"container/list"
	"hash/fnv"
	"sync/atomic"
	"time"
)

type Cache struct {
	*Configuration
	list        *list.List
	size        int64
	buckets     []*bucket
	bucketMask  uint32
	deletables  chan *Item
	promotables chan *Item
	donec       chan struct{}
}

// Create a new cache with the specified configuration
// See ccache.Configure() for creating a configuration
func New(config *Configuration) *Cache {
	c := &Cache{
		list:          list.New(),
		Configuration: config,
		bucketMask:    uint32(config.buckets) - 1,
		buckets:       make([]*bucket, config.buckets),
	}
	for i := 0; i < int(config.buckets); i++ {
		c.buckets[i] = &bucket{
			lookup: make(map[string]*Item),
		}
	}
	c.restart()
	return c
}

// Get an item from the cache. Returns nil if the item wasn't found.
// This can return an expired item. Use item.Expired() to see if the item
// is expired and item.TTL() to see how long until the item expires (which
// will be negative for an already expired item).
func (c *Cache) Get(key string) *Item {
	item := c.bucket(key).get(key)
	if item == nil {
		return nil
	}
	if item.expires > time.Now().UnixNano() {
		c.promote(item)
	}
	return item
}

// Used when the cache was created with the Track() configuration option.
// Avoid otherwise
func (c *Cache) TrackingGet(key string) TrackedItem {
	item := c.Get(key)
	if item == nil {
		return NilTracked
	}
	item.track()
	return item
}

// Set the value in the cache for the specified duration
func (c *Cache) Set(key string, value interface{}, duration time.Duration) {
	c.set(key, value, duration)
}

// Replace the value if it exists, does not set if it doesn't.
// Returns true if the item existed an was replaced, false otherwise.
// Replace does not reset item's TTL
func (c *Cache) Replace(key string, value interface{}) bool {
	item := c.bucket(key).get(key)
	if item == nil {
		return false
	}
	c.Set(key, value, item.TTL())
	return true
}

// Attempts to get the value from the cache and calles fetch on a miss (missing
// or stale item). If fetch returns an error, no value is cached and the error
// is returned back to the caller.
func (c *Cache) Fetch(key string, duration time.Duration, fetch func() (interface{}, error)) (*Item, error) {
	item := c.Get(key)
	if item != nil && !item.Expired() {
		return item, nil
	}
	value, err := fetch()
	if err != nil {
		return nil, err
	}
	return c.set(key, value, duration), nil
}

// Remove the item from the cache, return true if the item was present, false otherwise.
func (c *Cache) Delete(key string) bool {
	item := c.bucket(key).delete(key)
	if item != nil {
		c.deletables <- item
		return true
	}
	return false
}

//this isn't thread safe. It's meant to be called from non-concurrent tests
func (c *Cache) Clear() {
	for _, bucket := range c.buckets {
		bucket.clear()
	}
	c.size = 0
	c.list = list.New()
}

// Stops the background worker. Operations performed on the cache after Stop
// is called are likely to panic
func (c *Cache) Stop() {
	close(c.promotables)
	<-c.donec
}

func (c *Cache) restart() {
	c.deletables = make(chan *Item, c.deleteBuffer)
	c.promotables = make(chan *Item, c.promoteBuffer)
	c.donec = make(chan struct{})
	go c.worker()
}

func (c *Cache) deleteItem(bucket *bucket, item *Item) {
	bucket.delete(item.key) //stop other GETs from getting it
	c.deletables <- item
}

func (c *Cache) set(key string, value interface{}, duration time.Duration) *Item {
	item, existing := c.bucket(key).set(key, value, duration)
	if existing != nil {
		c.deletables <- existing
	}
	c.promote(item)
	return item
}

func (c *Cache) bucket(key string) *bucket {
	h := fnv.New32a()
	h.Write([]byte(key))
	return c.buckets[h.Sum32()&c.bucketMask]
}

func (c *Cache) promote(item *Item) {
	c.promotables <- item
}

func (c *Cache) worker() {
	defer close(c.donec)

	for {
		select {
		case item, ok := <-c.promotables:
			if ok == false {
				goto drain
			}
			if c.doPromote(item) && c.size > c.maxSize {
				c.gc()
			}
		case item := <-c.deletables:
			c.doDelete(item)
		}
	}

drain:
	for {
		select {
		case item := <-c.deletables:
			c.doDelete(item)
		default:
			close(c.deletables)
			return
		}
	}
}

func (c *Cache) doDelete(item *Item) {
	if item.element == nil {
		item.promotions = -2
	} else {
		c.size -= item.size
		c.list.Remove(item.element)
	}
}

func (c *Cache) doPromote(item *Item) bool {
	//already deleted
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

func (c *Cache) gc() {
	element := c.list.Back()
	for i := 0; i < c.itemsToPrune; i++ {
		if element == nil {
			return
		}
		prev := element.Prev()
		item := element.Value.(*Item)
		if c.tracking == false || atomic.LoadInt32(&item.refCount) == 0 {
			c.bucket(item.key).delete(item.key)
			c.size -= item.size
			c.list.Remove(element)
			item.promotions = -2
		}
		element = prev
	}
}
