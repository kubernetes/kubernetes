/*
Copyright 2016 The Kubernetes Authors.

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

package chaosmonkey

import (
	"context"
	"sync/atomic"
	"testing"
)

func TestDoWithPanic(t *testing.T) {
	var counter int64
	cm := New(func(ctx context.Context) {})
	tests := []Test{
		// No panic
		func(ctx context.Context, sem *Semaphore) {
			defer atomic.AddInt64(&counter, 1)
			sem.Ready()
		},
		// Panic after sem.Ready()
		func(ctx context.Context, sem *Semaphore) {
			defer atomic.AddInt64(&counter, 1)
			sem.Ready()
			panic("Panic after calling sem.Ready()")
		},
		// Panic before sem.Ready()
		func(ctx context.Context, sem *Semaphore) {
			defer atomic.AddInt64(&counter, 1)
			panic("Panic before calling sem.Ready()")
		},
	}
	for _, test := range tests {
		cm.Register(test)
	}
	cm.Do(context.Background())
	// Check that all funcs in tests were called.
	if int(counter) != len(tests) {
		t.Errorf("Expected counter to be %v, but it was %v", len(tests), counter)
	}
}
