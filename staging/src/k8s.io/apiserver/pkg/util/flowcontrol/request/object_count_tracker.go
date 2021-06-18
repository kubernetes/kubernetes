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

package request

import (
	"sync"
)

// StorageObjectCountTracker is an interface that is used to keep track of
// of the total number of objects for each resource.
// {group}.{resource} is used as the key name to update and retrieve
// the total number of objects for a given resource.
type StorageObjectCountTracker interface {
	// OnCount is invoked to update the current number of total
	// objects for the given resource
	OnCount(string, int64)

	// Get returns the total number of objects for the given resource.
	// If the given resource is not being tracked Get will return zero.
	// For now, we do not differentiate between zero object count and
	// a given resoure not being present.
	Get(string) int64
}

// NewStorageObjectCountTracker returns an instance of
// StorageObjectCountTracker interface that can be used to
// keep track of the total number of objects for each resource.
func NewStorageObjectCountTracker() StorageObjectCountTracker {
	return &objectCountTracker{
		counts: map[string]int64{},
	}
}

// objectCountTracker implements StorageObjectCountTracker with
// reader/writer mutual exclusion lock.
type objectCountTracker struct {
	lock   sync.RWMutex
	counts map[string]int64
}

func (t *objectCountTracker) OnCount(key string, count int64) {
	t.lock.Lock()
	defer t.lock.Unlock()

	t.counts[key] = count
}

func (t *objectCountTracker) Get(key string) int64 {
	t.lock.RLock()
	defer t.lock.RUnlock()

	return t.counts[key]
}
