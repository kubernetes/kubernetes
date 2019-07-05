package simplelru

// LRUCache is the interface for simple LRU cache.
type LRUCache interface {
	// Adds a value to the cache, returns true if an eviction occurred and
	// updates the "recently used"-ness of the key.
	Add(key, value interface{}) bool

	// Returns key's value from the cache and
	// updates the "recently used"-ness of the key. #value, isFound
	Get(key interface{}) (value interface{}, ok bool)

	// Check if a key exsists in cache without updating the recent-ness.
	Contains(key interface{}) (ok bool)

	// Returns key's value without updating the "recently used"-ness of the key.
	Peek(key interface{}) (value interface{}, ok bool)

	// Removes a key from the cache.
	Remove(key interface{}) bool

	// Removes the oldest entry from cache.
	RemoveOldest() (interface{}, interface{}, bool)

	// Returns the oldest entry from the cache. #key, value, isFound
	GetOldest() (interface{}, interface{}, bool)

	// Returns a slice of the keys in the cache, from oldest to newest.
	Keys() []interface{}

	// Returns the number of items in the cache.
	Len() int

	// Clear all cache entries
	Purge()
}
