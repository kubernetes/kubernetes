/*
Copyright 2019 The Kubernetes Authors.

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

package v1alpha1

import (
	"errors"
	"sync"
)

const (
	// NotFound is the not found error message.
	NotFound = "not found"
)

// ContextData is a generic type for arbitrary data stored in PluginContext.
type ContextData interface{}

// ContextKey is the type of keys stored in PluginContext.
type ContextKey string

// PluginContext provides a mechanism for plugins to store and retrieve arbitrary data.
// ContextData stored by one plugin can be read, altered, or deleted by another plugin.
// PluginContext does not provide any data protection, as all plugins are assumed to be
// trusted.
type PluginContext struct {
	mx      sync.RWMutex
	storage map[ContextKey]ContextData
}

// NewPluginContext initializes a new PluginContext and returns its pointer.
func NewPluginContext() *PluginContext {
	return &PluginContext{
		storage: make(map[ContextKey]ContextData),
	}
}

// Read retrieves data with the given "key" from PluginContext. If the key is not
// present an error is returned.
// This function is not thread safe. In multi-threaded code, lock should be
// acquired first.
func (c *PluginContext) Read(key ContextKey) (ContextData, error) {
	if v, ok := c.storage[key]; ok {
		return v, nil
	}
	return nil, errors.New(NotFound)
}

// Write stores the given "val" in PluginContext with the given "key".
// This function is not thread safe. In multi-threaded code, lock should be
// acquired first.
func (c *PluginContext) Write(key ContextKey, val ContextData) {
	c.storage[key] = val
}

// Delete deletes data with the given key from PluginContext.
// This function is not thread safe. In multi-threaded code, lock should be
// acquired first.
func (c *PluginContext) Delete(key ContextKey) {
	delete(c.storage, key)
}

// Lock acquires PluginContext lock.
func (c *PluginContext) Lock() {
	c.mx.Lock()
}

// Unlock releases PluginContext lock.
func (c *PluginContext) Unlock() {
	c.mx.Unlock()
}

// RLock acquires PluginContext read lock.
func (c *PluginContext) RLock() {
	c.mx.RLock()
}

// RUnlock releases PluginContext read lock.
func (c *PluginContext) RUnlock() {
	c.mx.RUnlock()
}
