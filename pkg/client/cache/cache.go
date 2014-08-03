/*
Copyright 2014 Google Inc. All rights reserved.

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

package cache

import (
	"sync"
)

type cache struct {
	lock  sync.RWMutex
	items map[string]interface{}
}

// Add inserts an item into the cache.
func (c *cache) Add(ID string, obj interface{}) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.items[ID] = obj
}

// Update sets an item in the cache to its updated state.
func (c *cache) Update(ID string, obj interface{}) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.items[ID] = obj
}

// Delete removes an item from the cache.
func (c *cache) Delete(ID string, obj interface{}) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.items, ID)
}

// List returns a list of all the items.
func (c *cache) List() []interface{} {
	c.lock.RLock()
	defer c.lock.RUnlock()
	list := make([]interface{}, 0, len(c.items))
	for _, item := range c.items {
		list = append(list, item)
	}
	return list
}

// Get returns the requested item, or sets exists=false.
func (c *cache) Get(ID string) (item interface{}, exists bool) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	item, exists = c.items[ID]
	return item, exists
}

// NewStore returns a Store implemented simply with a map and a lock.
func NewStore() Store {
	return &cache{items: map[string]interface{}{}}
}
