/*
Copyright The Kubernetes Authors.

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

package devicetainteviction

import (
	"time"

	"k8s.io/client-go/util/workqueue"
)

// NOPQueue is an implementation of [TypedRateLimitingInterface] which
// doesn't do anything.
type NOPQueue[T comparable] struct {
}

var _ workqueue.TypedRateLimitingInterface[int] = &NOPQueue[int]{}

// Add implements [TypedInterface].
func (m *NOPQueue[T]) Add(item T) {
}

// Len implements [TypedInterface].
func (m *NOPQueue[T]) Len() int {
	return 0
}

// Get implements [TypedInterface].
func (m *NOPQueue[T]) Get() (item T, shutdown bool) {
	shutdown = true
	return
}

// Done implements [TypedInterface].
func (m *NOPQueue[T]) Done(item T) {
}

// ShutDown implements [TypedInterface].
func (m *NOPQueue[T]) ShutDown() {
}

// ShutDownWithDrain implements [TypedInterface].
func (m *NOPQueue[T]) ShutDownWithDrain() {
}

// ShuttingDown implements [TypedInterface].
func (m *NOPQueue[T]) ShuttingDown() bool {
	return true
}

// AddAfter implements [TypedDelayingInterface.AddAfter]
func (m *NOPQueue[T]) AddAfter(item T, duration time.Duration) {
}

// AddRateLimited implements [TypedRateLimitingInterface.AddRateLimited].
func (m *NOPQueue[T]) AddRateLimited(item T) {
}

// Forget implements [TypedRateLimitingInterface.Forget].
func (m *NOPQueue[T]) Forget(item T) {
}

// NumRequeues implements [TypedRateLimitingInterface.NumRequeues].
func (m *NOPQueue[T]) NumRequeues(item T) int {
	return 0
}
