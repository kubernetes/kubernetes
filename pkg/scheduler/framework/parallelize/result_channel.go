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

package parallelize

import "context"

// ResultChannel supports non-blocking send and receive operation to capture result.
// A maximum of one result is kept in the channel and the rest of the results sent
// are ignored, unless the existing result is received and the channel becomes empty
// again.
type ResultChannel[T comparable] struct {
	ch chan T
}

// Send sends a result without blocking the sender.
func (e *ResultChannel[T]) Send(result T) {
	select {
	case e.ch <- result:
	default:
	}
}

// SendWithCancel sends a result without blocking the sender and calls
// cancel function.
func (e *ResultChannel[T]) SendWithCancel(result T, cancel context.CancelFunc) {
	e.Send(result)
	cancel()
}

// Receive receives a result from channel without blocking on the receiver.
func (e *ResultChannel[T]) Receive() T {
	select {
	case result := <-e.ch:
		return result
	default:
		var zeroValue T
		return zeroValue
	}
}

// NewResultChannel returns a new ResultChannel.
func NewResultChannel[T comparable]() *ResultChannel[T] {
	return &ResultChannel[T]{
		ch: make(chan T, 1),
	}
}
