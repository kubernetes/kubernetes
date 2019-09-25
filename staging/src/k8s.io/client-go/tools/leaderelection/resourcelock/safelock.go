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

package resourcelock

import (
	"sync"
)

var _ Interface = &safeLock{}

// safeLock is a wrapper for an existing resourcelock interface meant to be
// used by multiple goroutines.
type safeLock struct {
	base Interface
	mu   sync.Mutex
}

// Get returns the election record from a Lease spec
func (sl *safeLock) Get() (*LeaderElectionRecord, []byte, error) {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	return sl.base.Get()
}

// Create attempts to create a Lease
func (sl *safeLock) Create(ler LeaderElectionRecord) error {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	return sl.base.Create(ler)
}

// Update will update an existing Lease spec.
func (sl *safeLock) Update(ler LeaderElectionRecord) error {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	return sl.base.Update(ler)
}

// RecordEvent in leader election while adding meta-data
func (sl *safeLock) RecordEvent(s string) {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	sl.base.RecordEvent(s)
}

// Describe is used to convert details on current resource lock into a string
func (sl *safeLock) Describe() string {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	return sl.base.Describe()
}

// Identity returns the Identity of the lock
func (sl *safeLock) Identity() string {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	return sl.base.Identity()
}
