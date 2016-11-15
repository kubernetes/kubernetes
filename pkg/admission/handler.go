/*
Copyright 2015 The Kubernetes Authors.

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

package admission

import (
	"time"

	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	// timeToWaitForReady is the amount of time to wait to let an admission controller to be ready to satisfy a request.
	// this is useful when admission controllers need to warm their caches before letting requests through.
	timeToWaitForReady = 10 * time.Second
)

// ReadyFunc is a function that returns true if the admission controller is ready to handle requests.
type ReadyFunc func() bool

// Handler is a base for admission control handlers that
// support a predefined set of operations
type Handler struct {
	operations sets.String
	readyFunc  ReadyFunc
}

// Handles returns true for methods that this handler supports
func (h *Handler) Handles(operation Operation) bool {
	return h.operations.Has(string(operation))
}

// NewHandler creates a new base handler that handles the passed
// in operations
func NewHandler(ops ...Operation) *Handler {
	operations := sets.NewString()
	for _, op := range ops {
		operations.Insert(string(op))
	}
	return &Handler{
		operations: operations,
	}
}

// SetReadyFunc allows late registration of a ReadyFunc to know if the handler is ready to process requests.
func (h *Handler) SetReadyFunc(readyFunc ReadyFunc) {
	h.readyFunc = readyFunc
}

// WaitForReady will wait for the readyFunc (if registered) to return ready, and in case of timeout, will return false.
func (h *Handler) WaitForReady() bool {
	// there is no ready func configured, so we return immediately
	if h.readyFunc == nil {
		return true
	}
	return h.waitForReadyInternal(time.After(timeToWaitForReady))
}

func (h *Handler) waitForReadyInternal(timeout <-chan time.Time) bool {
	// there is no configured ready func, so return immediately
	if h.readyFunc == nil {
		return true
	}
	for !h.readyFunc() {
		select {
		case <-time.After(100 * time.Millisecond):
		case <-timeout:
			return h.readyFunc()
		}
	}
	return true
}
