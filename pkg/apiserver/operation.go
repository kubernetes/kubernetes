/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func init() {
	api.AddKnownTypes(ServerOp{}, ServerOpList{})
}

// Operation information, as delivered to API clients.
type ServerOp struct {
	api.JSONBase `yaml:",inline" json:",inline"`
}

// Operation list, as delivered to API clients.
type ServerOpList struct {
	api.JSONBase `yaml:",inline" json:",inline"`
	Items        []ServerOp `yaml:"items,omitempty" json:"items,omitempty"`
}

// Operation represents an ongoing action which the server is performing.
type Operation struct {
	ID       string
	result   interface{}
	awaiting <-chan interface{}
	finished *time.Time
	lock     sync.Mutex
	notify   chan bool
}

// Operations tracks all the ongoing operations.
type Operations struct {
	lock   sync.Mutex
	ops    map[string]*Operation
	nextID int
}

// Returns a new Operations repository.
func NewOperations() *Operations {
	ops := &Operations{
		ops: map[string]*Operation{},
	}
	go util.Forever(func() { ops.expire(10 * time.Minute) }, 5*time.Minute)
	return ops
}

// Add a new operation.
func (ops *Operations) NewOperation(from <-chan interface{}) *Operation {
	ops.lock.Lock()
	defer ops.lock.Unlock()
	id := fmt.Sprintf("%v", ops.nextID)
	ops.nextID++

	op := &Operation{
		ID:       id,
		awaiting: from,
		notify:   make(chan bool, 1),
	}
	go op.wait()
	ops.ops[id] = op
	return op
}

// List operations for an API client.
func (ops *Operations) List() ServerOpList {
	ops.lock.Lock()
	defer ops.lock.Unlock()

	ids := []string{}
	for id := range ops.ops {
		ids = append(ids, id)
	}
	sort.StringSlice(ids).Sort()
	ol := ServerOpList{}
	for _, id := range ids {
		ol.Items = append(ol.Items, ServerOp{JSONBase: api.JSONBase{ID: id}})
	}
	return ol
}

// Returns the operation with the given ID, or nil
func (ops *Operations) Get(id string) *Operation {
	ops.lock.Lock()
	defer ops.lock.Unlock()
	return ops.ops[id]
}

// Garbage collect operations that have finished longer than maxAge ago.
func (ops *Operations) expire(maxAge time.Duration) {
	ops.lock.Lock()
	defer ops.lock.Unlock()
	keep := map[string]*Operation{}
	limitTime := time.Now().Add(-maxAge)
	for id, op := range ops.ops {
		if !op.expired(limitTime) {
			keep[id] = op
		}
	}
	ops.ops = keep
}

// Waits forever for the operation to complete; call via go when
// the operation is created. Sets op.finished when the operation
// does complete. Does not keep op locked while waiting.
func (op *Operation) wait() {
	defer util.HandleCrash()
	result := <-op.awaiting

	op.lock.Lock()
	defer op.lock.Unlock()
	op.result = result
	finished := time.Now()
	op.finished = &finished
	op.notify <- true
}

// Wait for the specified duration, or until the operation finishes,
// whichever happens first.
func (op *Operation) WaitFor(timeout time.Duration) {
	select {
	case <-time.After(timeout):
	case <-op.notify:
		// Re-send on this channel in case there are others
		// waiting for notification.
		op.notify <- true
	}
}

// Returns true if this operation finished before limitTime.
func (op *Operation) expired(limitTime time.Time) bool {
	op.lock.Lock()
	defer op.lock.Unlock()
	if op.finished == nil {
		return false
	}
	return op.finished.Before(limitTime)
}

// Return status information or the result of the operation if it is complete,
// with a bool indicating true in the latter case.
func (op *Operation) Describe() (description interface{}, finished bool) {
	op.lock.Lock()
	defer op.lock.Unlock()

	if op.finished == nil {
		return api.Status{
			Status:  api.StatusWorking,
			Details: op.ID,
		}, false
	}
	return op.result, true
}
