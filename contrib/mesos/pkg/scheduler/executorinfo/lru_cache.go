/*
Copyright 2015 The Kubernetes Authors.

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

package executorinfo

import (
	"container/list"
	"errors"

	"github.com/mesos/mesos-go/mesosproto"
)

// Cache is an LRU cache for executor info objects.
// It is not safe for concurrent use.
type Cache struct {
	maxEntries int
	ll         *list.List
	cache      map[string]*list.Element // by hostname
}

type entry struct {
	hostname string
	info     *mesosproto.ExecutorInfo
}

// NewCache creates a new cache.
// If maxEntries is zero, an error is being returned.
func NewCache(maxEntries int) (*Cache, error) {
	if maxEntries <= 0 {
		return nil, errors.New("invalid maxEntries value")
	}

	return &Cache{
		maxEntries: maxEntries,
		ll:         list.New(), // least recently used sorted linked list
		cache:      make(map[string]*list.Element),
	}, nil
}

// Add adds an executor info associated with the given hostname to the cache.
func (c *Cache) Add(hostname string, e *mesosproto.ExecutorInfo) {
	if ee, ok := c.cache[hostname]; ok {
		c.ll.MoveToFront(ee)
		ee.Value.(*entry).info = e
		return
	}
	el := c.ll.PushFront(&entry{hostname, e})
	c.cache[hostname] = el
	if c.ll.Len() > c.maxEntries {
		c.RemoveOldest()
	}
}

// Get looks up a hostname's executor info from the cache.
func (c *Cache) Get(hostname string) (e *mesosproto.ExecutorInfo, ok bool) {
	if el, hit := c.cache[hostname]; hit {
		c.ll.MoveToFront(el)
		return el.Value.(*entry).info, true
	}
	return
}

// Remove removes the provided hostname from the cache.
func (c *Cache) Remove(hostname string) {
	if el, hit := c.cache[hostname]; hit {
		c.removeElement(el)
	}
}

// RemoveOldest removes the oldest item from the cache.
func (c *Cache) RemoveOldest() {
	oldest := c.ll.Back()
	if oldest != nil {
		c.removeElement(oldest)
	}
}

func (c *Cache) removeElement(el *list.Element) {
	c.ll.Remove(el)
	kv := el.Value.(*entry)
	delete(c.cache, kv.hostname)
}
