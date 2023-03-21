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

package executor

import (
	"fmt"
	"net/http"
	goruntime "runtime"
	"sync"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// Executor will execute the specified function in new goroutine(s), and
// return a channel (<-chan RecoveredPanic) that the caller can use to wait until
// the specified function has finished executing.
// If the value returned from the channel is nil, it indicates that the
// function has finished execution, and it did not throw any panic, otherwise
// the returned value represents a recovered panic with stack trace.
//
// Here is a simple use case:
//
//	func() {
//		ch := Execute(func() {
//			// do something
//		})
//
//		select {
//		case <-ctx.Done():
//		case ret := <-ch:
//			if ret != nil {
//				// the function panicked, rethrow it
//				panic(ret)
//			}
//		}
//	}()
//
// The Executor will handle any panic thrown by the given function:
//   - it will recover, if there is a panic it will save the panic error, and
//     will capture the stack trace for the panicking goroutine.
//   - The Executor will NOT raise the panic again in order to prevent the
//     application from crashing.
//   - The caller can use the channel (<-chan RecoveredPanic) to get an object
//     that will have the recovered panic error, and the stack trace.
//   - when execution finishes without any panic, the channel would not have
//     any RecoveredPanic value, and it would be in a closed state.
//
// The Executor is a useful abstraction when a request handler wants
// to spin up goroutine(s) to get work done concurrently.
type Executor interface {
	Execute(func()) <-chan RecoveredPanic
}

// RecoveredPanic is an object that encapsulates recovered panic(s),
// when stringified/printed it should provide the panic reason and
// the corresponding stack trace.
type RecoveredPanic interface {
	String() string
}

// Single returns an Executor that can be used to execute a
// user specified function concurrently on a single goroutine.
func Single() Executor { return single{} }

// Pool returns an Executor that can be used to execute a
// user specified function concurrently on n goroutine(s).
func Pool(n int32) Executor { return pool{n: n} }

type single struct{}

func (s single) Execute(fn func()) <-chan RecoveredPanic {
	// use a buffered channel here, we don't want the sender
	// (worker goroutine) to block forever, in case the
	// receiver (caller) is not waiting to read from it.
	ch := make(chan RecoveredPanic, 1)
	go func() {
		defer close(ch)
		defer func() {
			if recovered := recover(); recovered != nil {
				// the channel gets written to only when there is a panic
				ch <- capture(recovered)
			}
		}()

		fn()
	}()

	return ch
}

type pool struct {
	n int32
}

func (p pool) Execute(fn func()) <-chan RecoveredPanic {
	// in the worst case, the number of goroutine(s) that can panic is p.n,
	// so at most, p.n goroutine(s) will write to this channel.
	recoveredCh := make(chan RecoveredPanic, p.n)
	accumulator := accumulator{recoveredCh: recoveredCh}

	lock := sync.Mutex{}
	counter := p.n
	for i := 0; i < int(p.n); i++ {
		go func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					recoveredCh <- capture(recovered)
				}

				// the lock ensures that all writes to the channel happen
				// before the last goroutine closes the channel.
				lock.Lock()
				defer lock.Unlock()
				counter--
				if counter == 0 {
					close(recoveredCh)
				}
			}()

			fn()
		}()
	}

	// the accumulator will collect each recovered panic from
	// the individual goroutine(s), and then consolidate into
	// a single object. the caller needs to read from the channel
	// into which the consolidated panic information gets written to.
	return accumulator.accumulate()
}

type accumulator struct {
	recoveredCh <-chan RecoveredPanic
}

func (a accumulator) accumulate() <-chan RecoveredPanic {
	// use a buffered channel here, we don't want the sender
	// (this goroutine) to block forever, in case the
	// receiver (caller) is not waiting to read from it.
	ch := make(chan RecoveredPanic, 1)

	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				utilruntime.HandleError(fmt.Errorf("the accumulator panicked: %v", capture(recovered)))
			}
		}()
		defer close(ch)

		allRecoveredPanics := RecoveredPanics{}
		// we rely on the sender to close this channel once all the individual
		// ReasonAndStackTrace objects have been written to.
		for recovered := range a.recoveredCh {
			allRecoveredPanics = append(allRecoveredPanics, recovered)
		}

		switch {
		case len(allRecoveredPanics) == 1:
			ch <- allRecoveredPanics[0]
		case len(allRecoveredPanics) > 1:
			ch <- allRecoveredPanics
		}
	}()

	return ch
}

type ReasonAndStackTrace struct {
	// Reason is the error recovered from the panic
	Reason interface{}

	// StackTrace is the captured stack trace of the goroutine that panicked
	StackTrace string
}

func (rp ReasonAndStackTrace) String() string {
	return fmt.Sprintf("%v\n%s", rp.Reason, rp.StackTrace)
}

// RecoveredPanics consolidates multiple ReasonAndStackTrace objects into one
type RecoveredPanics []RecoveredPanic

func (rps RecoveredPanics) String() string {
	switch {
	case len(rps) == 1:
		return rps[0].String()
	case len(rps) > 1:
		msg := fmt.Sprintf("%d worker goroutine(s) have panicked", len(rps))
		for _, rp := range rps {
			msg = fmt.Sprintf("%s\n%s", msg, rp.String())
		}
		return msg
	}
	return ""
}

func capture(recovered interface{}) ReasonAndStackTrace {
	// store the panic reason into the result.
	r := ReasonAndStackTrace{
		Reason: recovered,
	}

	// do not wrap the sentinel ErrAbortHandler panic value
	if recovered != http.ErrAbortHandler {
		// Same as stdlib http server code. Manually allocate stack
		// trace buffer size to prevent excessively large logs
		const size = 64 << 10
		buf := make([]byte, size)
		buf = buf[:goruntime.Stack(buf, false)]

		r.StackTrace = fmt.Sprintf("%s", buf)
	}

	return r
}
