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

package framework

import (
	"errors"
	"sync"
)

var (
	// ErrNotFound is the not found error message.
	ErrNotFound = errors.New("not found")
)

// StateData is a generic type for arbitrary data stored in CycleState.
type StateData interface {
	// Clone is an interface to make a copy of StateData. For performance reasons,
	// clone should make shallow copies for members (e.g., slices or maps) that are not
	// impacted by PreFilter's optional AddPod/RemovePod methods.
	Clone() StateData
}

// StateKey is the type of keys stored in CycleState.
type StateKey string

// CycleState provides a mechanism for plugins to store and retrieve arbitrary data.
// StateData stored by one plugin can be read, altered, or deleted by another plugin.
// CycleState does not provide any data protection, as all plugins are assumed to be
// trusted.
type CycleState struct {
	mx      sync.RWMutex
	storage map[StateKey]StateData
	// if recordPluginMetrics is true, PluginExecutionDuration will be recorded for this cycle.
	recordPluginMetrics bool
}

// NewCycleState initializes a new CycleState and returns its pointer.
func NewCycleState() *CycleState {
	return &CycleState{
		storage: make(map[StateKey]StateData),
	}
}

// ShouldRecordPluginMetrics returns whether PluginExecutionDuration metrics should be recorded.
func (c *CycleState) ShouldRecordPluginMetrics() bool {
	if c == nil {
		return false
	}
	return c.recordPluginMetrics
}

// SetRecordPluginMetrics sets recordPluginMetrics to the given value.
func (c *CycleState) SetRecordPluginMetrics(flag bool) {
	if c == nil {
		return
	}
	c.recordPluginMetrics = flag
}

// Clone creates a copy of CycleState and returns its pointer. Clone returns
// nil if the context being cloned is nil.
func (c *CycleState) Clone() *CycleState {
	if c == nil {
		return nil
	}
	copy := NewCycleState()
	for k, v := range c.storage {
		copy.Write(k, v.Clone())
	}
	return copy
}

// Read retrieves data with the given "key" from CycleState. If the key is not
// present an error is returned.
// This function is not thread safe. In multi-threaded code, lock should be
// acquired first.
func (c *CycleState) Read(key StateKey) (StateData, error) {
	if v, ok := c.storage[key]; ok {
		return v, nil
	}
	return nil, ErrNotFound
}

// Write stores the given "val" in CycleState with the given "key".
// This function is not thread safe. In multi-threaded code, lock should be
// acquired first.
func (c *CycleState) Write(key StateKey, val StateData) {
	c.storage[key] = val
}

// Delete deletes data with the given key from CycleState.
// This function is not thread safe. In multi-threaded code, lock should be
// acquired first.
func (c *CycleState) Delete(key StateKey) {
	delete(c.storage, key)
}

// Lock acquires CycleState lock.
func (c *CycleState) Lock() {
	c.mx.Lock()
}

// Unlock releases CycleState lock.
func (c *CycleState) Unlock() {
	c.mx.Unlock()
}

// RLock acquires CycleState read lock.
func (c *CycleState) RLock() {
	c.mx.RLock()
}

// RUnlock releases CycleState read lock.
func (c *CycleState) RUnlock() {
	c.mx.RUnlock()
}
