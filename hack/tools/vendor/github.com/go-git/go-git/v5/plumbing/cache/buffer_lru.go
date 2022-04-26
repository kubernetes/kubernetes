package cache

import (
	"container/list"
	"sync"
)

// BufferLRU implements an object cache with an LRU eviction policy and a
// maximum size (measured in object size).
type BufferLRU struct {
	MaxSize FileSize

	actualSize FileSize
	ll         *list.List
	cache      map[int64]*list.Element
	mut        sync.Mutex
}

// NewBufferLRU creates a new BufferLRU with the given maximum size. The maximum
// size will never be exceeded.
func NewBufferLRU(maxSize FileSize) *BufferLRU {
	return &BufferLRU{MaxSize: maxSize}
}

// NewBufferLRUDefault creates a new BufferLRU with the default cache size.
func NewBufferLRUDefault() *BufferLRU {
	return &BufferLRU{MaxSize: DefaultMaxSize}
}

type buffer struct {
	Key   int64
	Slice []byte
}

// Put puts a buffer into the cache. If the buffer is already in the cache, it
// will be marked as used. Otherwise, it will be inserted. A buffers might
// be evicted to make room for the new one.
func (c *BufferLRU) Put(key int64, slice []byte) {
	c.mut.Lock()
	defer c.mut.Unlock()

	if c.cache == nil {
		c.actualSize = 0
		c.cache = make(map[int64]*list.Element, 1000)
		c.ll = list.New()
	}

	bufSize := FileSize(len(slice))
	if ee, ok := c.cache[key]; ok {
		oldBuf := ee.Value.(buffer)
		// in this case bufSize is a delta: new size - old size
		bufSize -= FileSize(len(oldBuf.Slice))
		c.ll.MoveToFront(ee)
		ee.Value = buffer{key, slice}
	} else {
		if bufSize > c.MaxSize {
			return
		}
		ee := c.ll.PushFront(buffer{key, slice})
		c.cache[key] = ee
	}

	c.actualSize += bufSize
	for c.actualSize > c.MaxSize {
		last := c.ll.Back()
		lastObj := last.Value.(buffer)
		lastSize := FileSize(len(lastObj.Slice))

		c.ll.Remove(last)
		delete(c.cache, lastObj.Key)
		c.actualSize -= lastSize
	}
}

// Get returns a buffer by its key. It marks the buffer as used. If the buffer
// is not in the cache, (nil, false) will be returned.
func (c *BufferLRU) Get(key int64) ([]byte, bool) {
	c.mut.Lock()
	defer c.mut.Unlock()

	ee, ok := c.cache[key]
	if !ok {
		return nil, false
	}

	c.ll.MoveToFront(ee)
	return ee.Value.(buffer).Slice, true
}

// Clear the content of this buffer cache.
func (c *BufferLRU) Clear() {
	c.mut.Lock()
	defer c.mut.Unlock()

	c.ll = nil
	c.cache = nil
	c.actualSize = 0
}
