/*
Copyright 2016 The Kubernetes Authors.

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

package maps

import (
	"sync"
)

// Counter counts the occurences of a item, it is thread safe.
type Counter struct {
	mu     sync.Mutex
	values map[string]int64
}

// NewCounter returns a new counter object which counts elements in a map.
func NewCounter() *Counter {
	return &Counter{
		values: make(map[string]int64),
	}
}

// Get returns the count for a given value.
func (c *Counter) Get(key string) int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.values[key]
}

// Incr increases the count for a given value by 1.
func (c *Counter) Incr(key string) int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.values[key]++
	return c.values[key]
}
