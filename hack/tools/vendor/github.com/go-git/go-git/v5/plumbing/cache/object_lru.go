package cache

import (
	"container/list"
	"sync"

	"github.com/go-git/go-git/v5/plumbing"
)

// ObjectLRU implements an object cache with an LRU eviction policy and a
// maximum size (measured in object size).
type ObjectLRU struct {
	MaxSize FileSize

	actualSize FileSize
	ll         *list.List
	cache      map[interface{}]*list.Element
	mut        sync.Mutex
}

// NewObjectLRU creates a new ObjectLRU with the given maximum size. The maximum
// size will never be exceeded.
func NewObjectLRU(maxSize FileSize) *ObjectLRU {
	return &ObjectLRU{MaxSize: maxSize}
}

// NewObjectLRUDefault creates a new ObjectLRU with the default cache size.
func NewObjectLRUDefault() *ObjectLRU {
	return &ObjectLRU{MaxSize: DefaultMaxSize}
}

// Put puts an object into the cache. If the object is already in the cache, it
// will be marked as used. Otherwise, it will be inserted. A single object might
// be evicted to make room for the new object.
func (c *ObjectLRU) Put(obj plumbing.EncodedObject) {
	c.mut.Lock()
	defer c.mut.Unlock()

	if c.cache == nil {
		c.actualSize = 0
		c.cache = make(map[interface{}]*list.Element, 1000)
		c.ll = list.New()
	}

	objSize := FileSize(obj.Size())
	key := obj.Hash()
	if ee, ok := c.cache[key]; ok {
		oldObj := ee.Value.(plumbing.EncodedObject)
		// in this case objSize is a delta: new size - old size
		objSize -= FileSize(oldObj.Size())
		c.ll.MoveToFront(ee)
		ee.Value = obj
	} else {
		if objSize > c.MaxSize {
			return
		}
		ee := c.ll.PushFront(obj)
		c.cache[key] = ee
	}

	c.actualSize += objSize
	for c.actualSize > c.MaxSize {
		last := c.ll.Back()
		if last == nil {
			c.actualSize = 0
			break
		}

		lastObj := last.Value.(plumbing.EncodedObject)
		lastSize := FileSize(lastObj.Size())

		c.ll.Remove(last)
		delete(c.cache, lastObj.Hash())
		c.actualSize -= lastSize
	}
}

// Get returns an object by its hash. It marks the object as used. If the object
// is not in the cache, (nil, false) will be returned.
func (c *ObjectLRU) Get(k plumbing.Hash) (plumbing.EncodedObject, bool) {
	c.mut.Lock()
	defer c.mut.Unlock()

	ee, ok := c.cache[k]
	if !ok {
		return nil, false
	}

	c.ll.MoveToFront(ee)
	return ee.Value.(plumbing.EncodedObject), true
}

// Clear the content of this object cache.
func (c *ObjectLRU) Clear() {
	c.mut.Lock()
	defer c.mut.Unlock()

	c.ll = nil
	c.cache = nil
	c.actualSize = 0
}
