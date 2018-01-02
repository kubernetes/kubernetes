// Package redis provides a redis interface for http caching.
package redis

import (
	"github.com/garyburd/redigo/redis"
	"github.com/gregjones/httpcache"
)

// cache is an implementation of httpcache.Cache that caches responses in a
// redis server.
type cache struct {
	redis.Conn
}

// cacheKey modifies an httpcache key for use in redis. Specifically, it
// prefixes keys to avoid collision with other data stored in redis.
func cacheKey(key string) string {
	return "rediscache:" + key
}

// Get returns the response corresponding to key if present.
func (c cache) Get(key string) (resp []byte, ok bool) {
	item, err := redis.Bytes(c.Do("GET", cacheKey(key)))
	if err != nil {
		return nil, false
	}
	return item, true
}

// Set saves a response to the cache as key.
func (c cache) Set(key string, resp []byte) {
	c.Do("SET", cacheKey(key), resp)
}

// Delete removes the response with key from the cache.
func (c cache) Delete(key string) {
	c.Do("DEL", cacheKey(key))
}

// NewWithClient returns a new Cache with the given redis connection.
func NewWithClient(client redis.Conn) httpcache.Cache {
	return cache{client}
}
