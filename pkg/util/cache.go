/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"sync"
)

const (
	shardsCount int = 32
)

type Cache []*cacheShard

func NewCache(maxSize int) Cache {
	cache := make(Cache, shardsCount)
	for i := 0; i < shardsCount; i++ {
		cache[i] = &cacheShard{
			items:   make(map[uint64]interface{}),
			maxSize: maxSize / shardsCount,
		}
	}
	return cache
}

func (c Cache) getShard(index uint64) *cacheShard {
	return c[index%uint64(shardsCount)]
}

// Returns true if object already existed, false otherwise.
func (c *Cache) Add(index uint64, obj interface{}) bool {
	return c.getShard(index).add(index, obj)
}

func (c *Cache) Get(index uint64) (obj interface{}, found bool) {
	return c.getShard(index).get(index)
}

type cacheShard struct {
	items map[uint64]interface{}
	sync.RWMutex
	maxSize int
}

// Returns true if object already existed, false otherwise.
func (s *cacheShard) add(index uint64, obj interface{}) bool {
	s.Lock()
	defer s.Unlock()
	_, isOverwrite := s.items[index]
	s.items[index] = obj
	if len(s.items) > s.maxSize {
		var randomKey uint64
		for randomKey = range s.items {
			break
		}
		delete(s.items, randomKey)
	}
	return isOverwrite
}

func (s *cacheShard) get(index uint64) (obj interface{}, found bool) {
	s.RLock()
	defer s.RUnlock()
	obj, found = s.items[index]
	return
}
