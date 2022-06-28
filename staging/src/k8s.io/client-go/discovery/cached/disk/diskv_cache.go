/*
Copyright 2022 The Kubernetes Authors.

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

// NOTE(negz): This file is copied from the upstream httpcache implementation,
// which is no longer maintained.
// https://github.com/gregjones/httpcache/blob/901d90/diskcache/diskcache.go

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"io"

	"github.com/peterbourgon/diskv"
)

// Cache is an implementation of httpcache.Cache that supplements the in-memory map with persistent storage
type Cache struct {
	d *diskv.Diskv
}

// Get returns the response corresponding to key if present
func (c *Cache) Get(key string) (resp []byte, ok bool) {
	key = keyToFilename(key)
	resp, err := c.d.Read(key)
	if err != nil {
		return []byte{}, false
	}
	return resp, true
}

// Set saves a response to the cache as key
func (c *Cache) Set(key string, resp []byte) {
	key = keyToFilename(key)
	c.d.WriteStream(key, bytes.NewReader(resp), true)
}

// Delete removes the response with key from the cache
func (c *Cache) Delete(key string) {
	key = keyToFilename(key)
	c.d.Erase(key)
}

func keyToFilename(key string) string {
	h := md5.New()
	io.WriteString(h, key)
	return hex.EncodeToString(h.Sum(nil))
}

// New returns a new Cache that will store files in basePath
func New(basePath string) *Cache {
	return &Cache{
		d: diskv.New(diskv.Options{
			BasePath:     basePath,
			CacheSizeMax: 100 * 1024 * 1024, // 100MB
		}),
	}
}

// NewWithDiskv returns a new Cache using the provided Diskv as underlying
// storage.
func NewWithDiskv(d *diskv.Diskv) *Cache {
	return &Cache{d}
}
