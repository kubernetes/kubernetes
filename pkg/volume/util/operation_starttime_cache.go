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

package util

import (
	"sync"
	"time"
)

// operationTimestamp stores the start time of an operation by a plugin
type operationTimestamp struct {
	pluginName string
	operation  string
	startTs    time.Time
}

func newOperationTimestamp(pluginName, operationName string) *operationTimestamp {
	return &operationTimestamp{
		pluginName: pluginName,
		operation:  operationName,
		startTs:    time.Now(),
	}
}

// operationStartTimeCache concurrent safe cache for operation start timestamps
type operationStartTimeCache struct {
	cache sync.Map
}

// NewOperationStartTimeCache creates a operation timestamp cache
func NewOperationStartTimeCache() OperationStartTimeCache {
	return &operationStartTimeCache{
		cache: sync.Map{},
	}
}

// OperationStartTimeCache interface that defines functions commonly needed for
// volume operation end to end latency/error count metrics reporting
type OperationStartTimeCache interface {
	// AddIfNotExist returns directly if there exists an entry with the key. Otherwise, it
	// creates a new operation timestamp using operationName, pluginName, and current timestamp
	// and stores the operation timestamp with the key
	AddIfNotExist(key, pluginName, operationName string)

	// Delete deletes a value for a key.
	Delete(key string)

	// Has returns a bool value indicates the existence of a key in the cache
	Has(key string) bool

	// Load retrieves information from the cache by the passed in key. If there exists no such entry in the cache or
	// the entry in the cache is not of operationTimestamp type, "ok" will be set to false and
	// all other returned values should NOT be relied upon.
	Load(key string) (pluginName, operationName string, startTime time.Time, ok bool)

	// UpdatePluginName updates the pluginName field of a cached operationTimestamp entry
	// returns true upon success
	UpdatePluginName(key, newPluginName string) bool
}

func (c *operationStartTimeCache) AddIfNotExist(key, pluginName, operationName string) {
	ts := newOperationTimestamp(pluginName, operationName)
	c.cache.LoadOrStore(key, ts)
}

func (c *operationStartTimeCache) Delete(key string) {
	c.cache.Delete(key)
}

func (c *operationStartTimeCache) Has(key string) bool {
	_, exists := c.cache.Load(key)
	return exists
}

func (c *operationStartTimeCache) loadEntry(key string) (*operationTimestamp, bool) {
	obj, ok := c.cache.Load(key)
	if !ok {
		return nil, ok
	}
	ts, ok := obj.(*operationTimestamp)
	return ts, ok
}

func (c *operationStartTimeCache) Load(key string) (pluginName, operationName string, startTime time.Time, ok bool) {
	ts, ok := c.loadEntry(key)
	if !ok {
		return "", "", time.Time{}, ok
	}
	return ts.pluginName, ts.operation, ts.startTs, ok
}

func (c *operationStartTimeCache) UpdatePluginName(key, newPluginName string) bool {
	ts, ok := c.loadEntry(key)
	if !ok || ts.pluginName == newPluginName {
		return false
	}
	ts.pluginName = newPluginName
	newTs := newOperationTimestamp(newPluginName, ts.operation)
	newTs.startTs = ts.startTs
	c.cache.Store(key, newTs)
	return true
}
