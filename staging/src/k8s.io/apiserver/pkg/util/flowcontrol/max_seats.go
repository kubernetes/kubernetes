/*
Copyright 2023 The Kubernetes Authors.

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

package flowcontrol

import (
	"sync"
)

// MaxSeatsTracker is used to track max seats allocatable per priority level from the work estimator
type MaxSeatsTracker interface {
	// GetMaxSeats returns the maximum seats a request should occupy for a given priority level.
	GetMaxSeats(priorityLevelName string) uint64

	// SetMaxSeats configures max seats for a priority level.
	SetMaxSeats(priorityLevelName string, maxSeats uint64)

	// ForgetPriorityLevel removes max seats tracking for a priority level.
	ForgetPriorityLevel(priorityLevelName string)
}

type maxSeatsTracker struct {
	sync.RWMutex

	maxSeats map[string]uint64
}

func NewMaxSeatsTracker() MaxSeatsTracker {
	return &maxSeatsTracker{
		maxSeats: make(map[string]uint64),
	}
}

func (m *maxSeatsTracker) GetMaxSeats(plName string) uint64 {
	m.RLock()
	defer m.RUnlock()

	return m.maxSeats[plName]
}

func (m *maxSeatsTracker) SetMaxSeats(plName string, maxSeats uint64) {
	m.Lock()
	defer m.Unlock()

	m.maxSeats[plName] = maxSeats
}

func (m *maxSeatsTracker) ForgetPriorityLevel(plName string) {
	m.Lock()
	defer m.Unlock()

	delete(m.maxSeats, plName)
}
