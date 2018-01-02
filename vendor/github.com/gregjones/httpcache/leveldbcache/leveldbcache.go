// Package leveldbcache provides an implementation of httpcache.Cache that
// uses github.com/syndtr/goleveldb/leveldb
package leveldbcache

import (
	"github.com/syndtr/goleveldb/leveldb"
)

// Cache is an implementation of httpcache.Cache with leveldb storage
type Cache struct {
	db *leveldb.DB
}

// Get returns the response corresponding to key if present
func (c *Cache) Get(key string) (resp []byte, ok bool) {
	var err error
	resp, err = c.db.Get([]byte(key), nil)
	if err != nil {
		return []byte{}, false
	}
	return resp, true
}

// Set saves a response to the cache as key
func (c *Cache) Set(key string, resp []byte) {
	c.db.Put([]byte(key), resp, nil)
}

// Delete removes the response with key from the cache
func (c *Cache) Delete(key string) {
	c.db.Delete([]byte(key), nil)
}

// New returns a new Cache that will store leveldb in path
func New(path string) (*Cache, error) {
	cache := &Cache{}

	var err error
	cache.db, err = leveldb.OpenFile(path, nil)

	if err != nil {
		return nil, err
	}
	return cache, nil
}

// NewWithDB returns a new Cache using the provided leveldb as underlying
// storage.
func NewWithDB(db *leveldb.DB) *Cache {
	return &Cache{db}
}
