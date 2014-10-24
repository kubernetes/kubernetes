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
	"net/http"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type OperationHandler struct {
	ops   *Operations
	codec runtime.Codec
}

func (h *OperationHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	parts := splitPath(req.URL.Path)
	if len(parts) > 1 || req.Method != "GET" {
		notFound(w, req)
		return
	}
	if len(parts) == 0 {
		// List outstanding operations.
		list := h.ops.List()
		writeJSON(http.StatusOK, h.codec, list, w)
		return
	}

	op := h.ops.Get(parts[0])
	if op == nil {
		notFound(w, req)
		return
	}

	result, complete := op.StatusOrResult()
	obj := result.Object
	if complete {
		writeJSON(http.StatusOK, h.codec, obj, w)
	} else {
		writeJSON(http.StatusAccepted, h.codec, obj, w)
	}
}

// Operation represents an ongoing action which the server is performing.
type Operation struct {
	ID        string
	result    RESTResult
	onReceive func(RESTResult)
	awaiting  <-chan RESTResult
	finished  *time.Time
	lock      sync.Mutex
	notify    chan struct{}
}

// Operations tracks all the ongoing operations.
type Operations struct {
	// Access only using functions from atomic.
	lastID int64

	// 'lock' guards the ops map.
	lock sync.Mutex
	ops  map[string]*Operation
}

// NewOperations returns a new Operations repository.
func NewOperations() *Operations {
	ops := &Operations{
		ops: map[string]*Operation{},
	}
	go util.Forever(func() { ops.expire(10 * time.Minute) }, 5*time.Minute)
	return ops
}

// NewOperation adds a new operation. It is lock-free. 'onReceive' will be called
// with the value read from 'from', when it is read.
func (ops *Operations) NewOperation(from <-chan RESTResult, onReceive func(RESTResult)) *Operation {
	id := atomic.AddInt64(&ops.lastID, 1)
	op := &Operation{
		ID:        strconv.FormatInt(id, 10),
		awaiting:  from,
		onReceive: onReceive,
		notify:    make(chan struct{}),
	}
	go op.wait()
	go ops.insert(op)
	return op
}

// insert inserts op into the ops map.
func (ops *Operations) insert(op *Operation) {
	ops.lock.Lock()
	defer ops.lock.Unlock()
	ops.ops[op.ID] = op
}

// List lists operations for an API client.
func (ops *Operations) List() *api.ServerOpList {
	ops.lock.Lock()
	defer ops.lock.Unlock()

	ids := []string{}
	for id := range ops.ops {
		ids = append(ids, id)
	}
	sort.StringSlice(ids).Sort()
	ol := &api.ServerOpList{}
	for _, id := range ids {
		ol.Items = append(ol.Items, api.ServerOp{ObjectMeta: api.ObjectMeta{Name: id}})
	}
	return ol
}

// Get returns the operation with the given ID, or nil.
func (ops *Operations) Get(id string) *Operation {
	ops.lock.Lock()
	defer ops.lock.Unlock()
	return ops.ops[id]
}

// expire garbage collect operations that have finished longer than maxAge ago.
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

// wait waits forever for the operation to complete; call via go when
// the operation is created. Sets op.finished when the operation
// does complete, and closes the notify channel, in case there
// are any WaitFor() calls in progress.
// Does not keep op locked while waiting.
func (op *Operation) wait() {
	defer util.HandleCrash()
	result := <-op.awaiting

	op.lock.Lock()
	defer op.lock.Unlock()
	if op.onReceive != nil {
		op.onReceive(result)
	}
	op.result = result
	finished := time.Now()
	op.finished = &finished
	close(op.notify)
}

// WaitFor waits for the specified duration, or until the operation finishes,
// whichever happens first.
func (op *Operation) WaitFor(timeout time.Duration) {
	select {
	case <-time.After(timeout):
	case <-op.notify:
	}
}

// expired returns true if this operation finished before limitTime.
func (op *Operation) expired(limitTime time.Time) bool {
	op.lock.Lock()
	defer op.lock.Unlock()
	if op.finished == nil {
		return false
	}
	return op.finished.Before(limitTime)
}

// StatusOrResult returns status information or the result of the operation if it is complete,
// with a bool indicating true in the latter case.
func (op *Operation) StatusOrResult() (description RESTResult, finished bool) {
	op.lock.Lock()
	defer op.lock.Unlock()

	if op.finished == nil {
		return RESTResult{Object: &api.Status{
			Status:  api.StatusWorking,
			Reason:  api.StatusReasonWorking,
			Details: &api.StatusDetails{ID: op.ID, Kind: "operation"},
		}}, false
	}
	return op.result, true
}
