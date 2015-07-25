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

package operationmanager

import (
	"fmt"
	"sync"
)

// Operation Manager is a thread-safe interface for keeping track of multiple pending async operations.
type OperationManager interface {
	// Called when the operation with the given ID has started.
	// Creates a new channel with specified buffer size tracked with the specified ID.
	// Returns a read-only version of the newly created channel.
	// Returns an error if an entry with the specified ID already exists (previous entry must be removed by calling Close).
	Start(id string, bufferSize uint) (<-chan interface{}, error)

	// Called when the operation with the given ID has terminated.
	// Closes and removes the channel associated with ID.
	// Returns an error if no associated channel exists.
	Close(id string) error

	// Attempts to send msg to the channel associated with ID.
	// Returns an error if no associated channel exists.
	Send(id string, msg interface{}) error

	// Returns true if an entry with the specified ID already exists.
	Exists(id string) bool
}

// Returns a new instance of a channel manager.
func NewOperationManager() OperationManager {
	return &operationManager{
		chanMap: make(map[string]chan interface{}),
	}
}

type operationManager struct {
	sync.RWMutex
	chanMap map[string]chan interface{}
}

// Called when the operation with the given ID has started.
// Creates a new channel with specified buffer size tracked with the specified ID.
// Returns a read-only version of the newly created channel.
// Returns an error if an entry with the specified ID already exists (previous entry must be removed by calling Close).
func (cm *operationManager) Start(id string, bufferSize uint) (<-chan interface{}, error) {
	cm.Lock()
	defer cm.Unlock()
	if _, exists := cm.chanMap[id]; exists {
		return nil, fmt.Errorf("id %q already exists", id)
	}
	cm.chanMap[id] = make(chan interface{}, bufferSize)
	return cm.chanMap[id], nil
}

// Called when the operation with the given ID has terminated.
// Closes and removes the channel associated with ID.
// Returns an error if no associated channel exists.
func (cm *operationManager) Close(id string) error {
	cm.Lock()
	defer cm.Unlock()
	if _, exists := cm.chanMap[id]; !exists {
		return fmt.Errorf("id %q not found", id)
	}
	close(cm.chanMap[id])
	delete(cm.chanMap, id)
	return nil
}

// Attempts to send msg to the channel associated with ID.
// Returns an error if no associated channel exists.
func (cm *operationManager) Send(id string, msg interface{}) error {
	cm.RLock()
	defer cm.RUnlock()
	if _, exists := cm.chanMap[id]; !exists {
		return fmt.Errorf("id %q not found", id)
	}
	cm.chanMap[id] <- msg
	return nil
}

// Returns true if an entry with the specified ID already exists.
func (cm *operationManager) Exists(id string) (exists bool) {
	cm.RLock()
	defer cm.RUnlock()
	_, exists = cm.chanMap[id]
	return
}
