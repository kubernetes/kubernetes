/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package component

import (
	"sync"

	"k8s.io/kubernetes/pkg/api"
)

// Cache maintains the readiness information(probe results) of
// components over time to allow for implementation of health thresholds.
// This manager is thread-safe, no locks are necessary for the caller.
type Cache struct {
	// guards states
	sync.RWMutex
	components map[string]api.Component
}

// NewCache creates ane returns a readiness manager with empty
// contents.
func NewCache() *Cache {
	return &Cache{components: make(map[string]api.Component)}
}

// Create adds the component to the cache
func (c *Cache) Create(id string, component api.Component) (created bool) {
	c.Lock()
	defer c.Unlock()
	_, found := c.components[id]
	if found {
		return false
	}
	c.components[id] = component
	return true
}

// Read returns a the component, if it is cached.
func (c *Cache) Read(id string) (api.Component, bool) {
	c.RLock()
	defer c.RUnlock()
	component, found := c.components[id]
	return component, found
}

// Create adds a copy of the component to the cache
func (c *Cache) Update(id string, component api.Component) (updated bool) {
	c.Lock()
	defer c.Unlock()
	_, found := c.components[id]
	if !found {
		return false
	}
	c.components[id] = component
	return true
}

// Delete removes the component from the cache
func (c *Cache) Delete(id string) {
	c.Lock()
	defer c.Unlock()
	delete(c.components, id)
}
