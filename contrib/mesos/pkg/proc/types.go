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

package proc

import (
	"time"
)

// Action is something that executes in the context of a process
type Action func()

type Context interface {
	// end (terminate) the execution context
	End() <-chan struct{}

	// return a signal chan that will close upon the termination of this process
	Done() <-chan struct{}
}

type Doer interface {
	// execute some action in some context. actions are to be executed in a
	// concurrency-safe manner: no two actions should execute at the same time.
	// errors are generated if the action cannot be executed (not by the execution
	// of the action) and should be testable with the error API of this package,
	// for example, IsProcessTerminated.
	Do(Action) <-chan error
}

// DoerFunc is an adapter func for Doer interface
type DoerFunc func(Action) <-chan error

// invoke the f on action a. returns an illegal state error if f is nil.
func (f DoerFunc) Do(a Action) <-chan error {
	if f != nil {
		return f(a)
	}
	return ErrorChan(errIllegalState)
}

type Process interface {
	Context
	Doer

	// see top level OnError func. this implementation will terminate upon the arrival of
	// an error (and subsequently invoke the error handler, if given) or else the termination
	// of the process (testable via IsProcessTerminated).
	OnError(<-chan error, func(error)) <-chan struct{}

	// return a signal chan that will close once the process is ready to run actions
	Running() <-chan struct{}
}

// ErrorOnce an error promise. If we ever start building out support for other promise types it will probably
// make sense to group them in some sort of "promises" package.
type ErrorOnce interface {
	// Err returns a chan that only ever sends one error, either obtained via Report() or Forward().
	Err() <-chan error

	// Report reports the given error via Err(), but only if no other errors have been reported or forwarded.
	Report(error)

	// Report reports an error via Err(), but only if no other errors have been reported or forwarded, using
	// fmt.Errorf to generate the error.
	Reportf(string, ...interface{})

	// forward waits for an error on the incoming chan, the result of which is later obtained via Err() (if no
	// other errors have been reported or forwarded).
	forward(<-chan error)

	// Send is non-blocking; it spins up a goroutine that reports an error (if any) that occurs on the error chan.
	Send(<-chan error) ErrorOnce

	// WaitFor returns true if an error is received within the specified duration, otherwise false
	WaitFor(time.Duration) (error, bool)
}
