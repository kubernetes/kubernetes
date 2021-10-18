/*
Copyright 2021 The Kubernetes Authors.

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

package disk

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"sync"

	"github.com/peterbourgon/diskv"
)

// memDiskCache is a reimplementation of github.com/gregjones/httpcache/diskcache.Cache that caches data in memory
// if there was an error persisting it to the diskv cache.
type memDiskCache struct {
	mu    sync.Mutex
	cache map[string][]byte
	d     *diskv.Diskv
}

// Get returns the response corresponding to key if present
func (c *memDiskCache) Get(key string) (resp []byte, ok bool) {
	key = keyToFilename(key)
	resp, err := c.d.Read(key)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		resp, ok := c.cache[key]
		return resp, ok
	} else {
		delete(c.cache, key)
	}
	return resp, true
}

// Set saves a response to the cache as key
func (c *memDiskCache) Set(key string, resp []byte) {
	key = keyToFilename(key)
	err := c.d.WriteStream(key, bytes.NewReader(resp), true)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		c.cache[key] = resp
	} else {
		delete(c.cache, key)
	}
}

// Delete removes the response with key from the cache
func (c *memDiskCache) Delete(key string) {
	key = keyToFilename(key)
	_ = c.d.Erase(key)
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.cache, key)
}

func keyToFilename(key string) string {
	md5sum := md5.Sum([]byte(key))
	return hex.EncodeToString(md5sum[:])
}

// newMemDiskCache returns a new memDiskCache using the provided Diskv as underlying
// storage.
func newMemDiskCache(d *diskv.Diskv) *memDiskCache {
	return &memDiskCache{
		cache: make(map[string][]byte),
		d:     d,
	}
}
