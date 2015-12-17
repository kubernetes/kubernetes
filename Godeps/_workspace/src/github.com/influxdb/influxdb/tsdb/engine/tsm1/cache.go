package tsm1

import (
	"fmt"
	"log"
	"os"
	"sort"
	"sync"
)

var ErrCacheMemoryExceeded = fmt.Errorf("cache maximum memory size exceeded")
var ErrCacheInvalidCheckpoint = fmt.Errorf("invalid checkpoint")

// entry is a set of values and some metadata.
type entry struct {
	values   Values // All stored values.
	needSort bool   // true if the values are out of order and require deduping.
}

// newEntry returns a new instance of entry.
func newEntry() *entry {
	return &entry{}
}

// add adds the given values to the entry.
func (e *entry) add(values []Value) {
	// if there are existing values make sure they're all less than the first of
	// the new values being added
	l := len(e.values)
	if l != 0 {
		lastValTime := e.values[l-1].UnixNano()
		if lastValTime >= values[0].UnixNano() {
			e.needSort = true
		}
	}
	e.values = append(e.values, values...)

	// if there's only one value, we know it's sorted
	if len(values) <= 1 {
		return
	}

	// make sure the new values were in sorted order
	min := values[0].UnixNano()
	for _, v := range values[1:] {
		if min >= v.UnixNano() {
			e.needSort = true
			break
		}
	}
}

// deduplicate sorts and orders the entry's values. If values are already deduped and
// and sorted, the function does no work and simply returns.
func (e *entry) deduplicate() {
	if !e.needSort || len(e.values) == 0 {
		return
	}
	e.values = e.values.Deduplicate()
	e.needSort = false
}

// Cache maintains an in-memory store of Values for a set of keys.
type Cache struct {
	mu      sync.RWMutex
	store   map[string]*entry
	size    uint64
	maxSize uint64

	// snapshots are the cache objects that are currently being written to tsm files
	// they're kept in memory while flushing so they can be queried along with the cache.
	// they are read only and should never be modified
	snapshots     []*Cache
	snapshotsSize uint64
}

// NewCache returns an instance of a cache which will use a maximum of maxSize bytes of memory.
func NewCache(maxSize uint64) *Cache {
	return &Cache{
		maxSize: maxSize,
		store:   make(map[string]*entry),
	}
}

// Write writes the set of values for the key to the cache. This function is goroutine-safe.
// It returns an error if the cache has exceeded its max size.
func (c *Cache) Write(key string, values []Value) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Enough room in the cache?
	newSize := c.size + uint64(Values(values).Size())
	if c.maxSize > 0 && newSize+c.snapshotsSize > c.maxSize {
		return ErrCacheMemoryExceeded
	}

	c.write(key, values)
	c.size = newSize

	return nil
}

// WriteMulti writes the map of keys and associated values to the cache. This function is goroutine-safe.
// It returns an error if the cache has exceeded its max size.
func (c *Cache) WriteMulti(values map[string][]Value) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	totalSz := 0
	for _, v := range values {
		totalSz += Values(v).Size()
	}

	// Enough room in the cache?
	newSize := c.size + uint64(totalSz)
	if c.maxSize > 0 && newSize+c.snapshotsSize > c.maxSize {
		return ErrCacheMemoryExceeded
	}

	for k, v := range values {
		c.write(k, v)
	}
	c.size = newSize

	return nil
}

// Snapshot will take a snapshot of the current cache, add it to the slice of caches that
// are being flushed, and reset the current cache with new values
func (c *Cache) Snapshot() *Cache {
	c.mu.Lock()
	defer c.mu.Unlock()

	snapshot := NewCache(c.maxSize)
	snapshot.store = c.store
	snapshot.size = c.size

	c.store = make(map[string]*entry)
	c.size = 0

	c.snapshots = append(c.snapshots, snapshot)
	c.snapshotsSize += snapshot.size

	// sort the snapshot before returning it. The compactor and any queries
	// coming in while it writes will need the values sorted
	for _, e := range snapshot.store {
		e.deduplicate()
	}

	return snapshot
}

// ClearSnapshot will remove the snapshot cache from the list of flushing caches and
// adjust the size
func (c *Cache) ClearSnapshot(snapshot *Cache) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i, cache := range c.snapshots {
		if cache == snapshot {
			c.snapshots = append(c.snapshots[:i], c.snapshots[i+1:]...)
			c.snapshotsSize -= snapshot.size
			break
		}
	}
}

// Size returns the number of point-calcuated bytes the cache currently uses.
func (c *Cache) Size() uint64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.size
}

// MaxSize returns the maximum number of bytes the cache may consume.
func (c *Cache) MaxSize() uint64 {
	return c.maxSize
}

// Keys returns a sorted slice of all keys under management by the cache.
func (c *Cache) Keys() []string {
	var a []string
	for k, _ := range c.store {
		a = append(a, k)
	}
	sort.Strings(a)
	return a
}

// Values returns a copy of all values, deduped and sorted, for the given key.
func (c *Cache) Values(key string) Values {
	c.mu.RLock()
	e := c.store[key]
	if e != nil && e.needSort {
		// Sorting is needed, so unlock and run the merge operation with
		// a write-lock. It is actually possible that the data will be
		// sorted by the time the merge runs, which would mean very occasionally
		// a write-lock will be held when only a read-lock is required.
		c.mu.RUnlock()
		return func() Values {
			c.mu.Lock()
			defer c.mu.Unlock()
			return c.merged(key)
		}()
	}

	// No sorting required for key, so just merge while continuing to hold read-lock.
	return func() Values {
		defer c.mu.RUnlock()
		return c.merged(key)
	}()
}

// Delete will remove the keys from the cache
func (c *Cache) Delete(keys []string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, k := range keys {
		delete(c.store, k)
	}
}

// merged returns a copy of hot and snapshot values. The copy will be merged, deduped, and
// sorted. It assumes all necessary locks have been taken. If the caller knows that the
// the hot source data for the key will not be changed, it is safe to call this function
// with a read-lock taken. Otherwise it must be called with a write-lock taken.
func (c *Cache) merged(key string) Values {
	e := c.store[key]
	if e == nil {
		if len(c.snapshots) == 0 {
			// No values in hot cache or snapshots.
			return nil
		}
	} else {
		e.deduplicate()
	}

	// Build the sequence of entries that will be returned, in the correct order.
	// Calculate the required size of the destination buffer.
	var entries []*entry
	sz := 0
	for _, s := range c.snapshots {
		e := s.store[key]
		if e != nil {
			entries = append(entries, e)
			sz += len(e.values)
		}
	}
	if e != nil {
		entries = append(entries, e)
		sz += len(e.values)
	}

	// Any entries? If not, return.
	if sz == 0 {
		return nil
	}

	// Create the buffer, and copy all hot values and snapshots. Individual
	// entries are sorted at this point, so now the code has to check if the
	// resultant buffer will be sorted from start to finish.
	var needSort bool
	values := make(Values, sz)
	n := 0
	for _, e := range entries {
		if !needSort && n > 0 {
			needSort = values[n-1].UnixNano() > e.values[0].UnixNano()
		}
		n += copy(values[n:], e.values)
	}

	if needSort {
		values = values.Deduplicate()
	}

	return values
}

// Store returns the underlying cache store. This is not goroutine safe!
// Protect access by using the Lock and Unlock functions on Cache.
func (c *Cache) Store() map[string]*entry {
	return c.store
}

func (c *Cache) Lock() {
	c.mu.Lock()
}

func (c *Cache) Unlock() {
	c.mu.Unlock()
}

// values returns the values for the key. It doesn't lock and assumes the data is
// already sorted. Should only be used in compact.go in the CacheKeyIterator
func (c *Cache) values(key string) Values {
	e := c.store[key]
	if e == nil {
		return nil
	}
	return e.values
}

// write writes the set of values for the key to the cache. This function assumes
// the lock has been taken and does not enforce the cache size limits.
func (c *Cache) write(key string, values []Value) {
	e, ok := c.store[key]
	if !ok {
		e = newEntry()
		c.store[key] = e
	}
	e.add(values)
}

// CacheLoader processes a set of WAL segment files, and loads a cache with the data
// contained within those files.  Processing of the supplied files take place in the
// order they exist in the files slice.
type CacheLoader struct {
	files []string

	Logger *log.Logger
}

// NewCacheLoader returns a new instance of a CacheLoader.
func NewCacheLoader(files []string) *CacheLoader {
	return &CacheLoader{
		files:  files,
		Logger: log.New(os.Stderr, "[cacheloader] ", log.LstdFlags),
	}
}

// Load returns a cache loaded with the data contained within the segment files.
// If, during reading of a segment file, corruption is encountered, that segment
// file is truncated up to and including the last valid byte, and processing
// continues with the next segment file.
func (cl *CacheLoader) Load(cache *Cache) error {
	for _, fn := range cl.files {
		if err := func() error {
			f, err := os.OpenFile(fn, os.O_CREATE|os.O_RDWR, 0666)
			if err != nil {
				return err
			}

			// Log some information about the segments.
			stat, err := os.Stat(f.Name())
			if err != nil {
				return err
			}
			cl.Logger.Printf("reading file %s, size %d", f.Name(), stat.Size())

			r := NewWALSegmentReader(f)
			defer r.Close()

			for r.Next() {
				entry, err := r.Read()
				if err != nil {
					n := r.Count()
					cl.Logger.Printf("file %s corrupt at position %d, truncating", f.Name(), n)
					if err := f.Truncate(n); err != nil {
						return err
					}
					break
				}

				switch t := entry.(type) {
				case *WriteWALEntry:
					if err := cache.WriteMulti(t.Values); err != nil {
						return err
					}
				case *DeleteWALEntry:
					cache.Delete(t.Keys)
				}
			}

			return nil
		}(); err != nil {
			return err
		}
	}
	return nil
}
