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

package waitgroup

import (
	"sync"
)

// OptionalWaitGroup is a WaitGroup or nothing.  Use it to track when
// goroutines start or stop or begin or end waiting if the pointer is
// not nil.  This abstraction is intended to make it easier to write a
// package that can be unit tested by a function that advances a fake
// clock once all the available work is done.  Real clients set the
// pointer to nil because they do not need this tracking.
type OptionalWaitGroup struct {
	WG *sync.WaitGroup
}

// NoWaitGroup constructs an OptionalWaitGroup with a nil pointer
func NoWaitGroup() OptionalWaitGroup {
	return OptionalWaitGroup{WG: nil}
}

// WrapWaitGroupPointer is the abstraction function
func WrapWaitGroupPointer(wg *sync.WaitGroup) OptionalWaitGroup {
	return OptionalWaitGroup{WG: wg}
}

// Increment adds one to the count of active goroutines if and only if the pointer is not nil
func (owg OptionalWaitGroup) Increment() {
	if owg.WG != nil {
		owg.WG.Add(1)
	}
}

// Decrement subtracts one from the count of active goroutines if and only if the pointer is not nil
func (owg OptionalWaitGroup) Decrement() {
	if owg.WG != nil {
		owg.WG.Add(-1)
	}
}
