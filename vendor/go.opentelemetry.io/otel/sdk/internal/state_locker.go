// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// StateLocker implements a two state lock algorithm that enabled lock free operations inside a state
// and a global lock for switching between states. At every time, only one state is active and one cold state.
// States are represented by int numbers 0 and 1.
//
// This was inspired by the algorithm used on the prometheus client library that can be found at:
// https://github.com/prometheus/client_golang/blob/e7776d2c54305c1b62fdb113b5a7e9b944c5c27e/prometheus/histogram.go#L227
//
// To execute operations within the same state, call `Start()` before the operation and call `End(idx)`
// to end this operation. The `idx` argument of `End()` is the index of the active state when the operation
// started and it is returned by the `Start()` method. It is recommended to defer the call to `End(idx)`.
//
// One can change the active state by calling `SwapActiveState(fn)`. `fn` is a function that will be executed *before*
// switching the active state. Operations such as preparing the new state shall be called by this function. This will
// wait in-flight operations to end.
//
// Example workflow:
// 1. State 0 is active.
// 1.1 Operations to the active state can happen with `Start()` and `End(idx)` methods.
// 2. Call to `SwitchState(fn)`
// 2.1 run `fn` function to prepare the new state
// 2.2 make state 1 active
// 2.3 wait in-flight operations of the state 0 to end.
// 3. State 1 is now active and every new operation are executed in it.
//
// `SwitchState(fn)` are synchronized with a mutex that can be access with the `Lock()` and `Unlock()` methods.
// Access to the cold state must also be synchronized to ensure the cold state is not in the middle of state switch
// since that could represent an invalid state.
//
type StateLocker struct {
	countsAndActiveIdx uint64
	finishedOperations [2]uint64

	sync.Mutex
}

// Start an operation that will happen on a state. The current active state is returned.
// A call to `End(idx int)` must happens for every `Start()` call.
func (c *StateLocker) Start() int {
	n := atomic.AddUint64(&c.countsAndActiveIdx, 1)
	return int(n >> 63)
}

// End an operation that happened to the idx state.
func (c *StateLocker) End(idx int) {
	atomic.AddUint64(&c.finishedOperations[idx], 1)
}

// ColdIdx returns the index of the cold state.
func (c *StateLocker) ColdIdx() int {
	return int((^c.countsAndActiveIdx) >> 63)
}

// SwapActiveState swaps the cold and active states.
//
// This will wait all for in-flight operations that are happening to the current
// active state to end, this ensure that all access to this state will be consistent.
//
// This is synchronized by a mutex.
func (c *StateLocker) SwapActiveState(beforeFn func()) {
	c.Lock()
	defer c.Unlock()

	if beforeFn != nil {
		// prepare the state change
		beforeFn()
	}

	// Adding 1<<63 switches the active index (from 0 to 1 or from 1 to 0)
	// without touching the count bits.
	n := atomic.AddUint64(&c.countsAndActiveIdx, 1<<63)

	// count represents how many operations have started *before* the state change.
	count := n & ((1 << 63) - 1)

	activeFinishedOperations := &c.finishedOperations[n>>63]
	// coldFinishedOperations are the number of operations that have *ended* on the previous state.
	coldFinishedOperations := &c.finishedOperations[(^n)>>63]

	// Await all cold writers to finish writing, when coldFinishedOperations == count, all in-flight operations
	// have finished and we can cleanly end the state change.
	for count != atomic.LoadUint64(coldFinishedOperations) {
		runtime.Gosched() // Let observations get work done.
	}

	// Make sure that the new state keeps the same count of *ended* operations.
	atomic.AddUint64(activeFinishedOperations, count)
	atomic.StoreUint64(coldFinishedOperations, 0)
}
