// Copyright 2010 Google Inc.
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

// GoMock - a mock framework for Go.
//
// Standard usage:
//   (1) Define an interface that you wish to mock.
//         type MyInterface interface {
//           SomeMethod(x int64, y string)
//         }
//   (2) Use mockgen to generate a mock from the interface.
//   (3) Use the mock in a test:
//         func TestMyThing(t *testing.T) {
//           mockCtrl := gomock.NewController(t)
//           defer mockCtrl.Finish()
//
//           mockObj := something.NewMockMyInterface(mockCtrl)
//           mockObj.EXPECT().SomeMethod(4, "blah")
//           // pass mockObj to a real object and play with it.
//         }
//
// By default, expected calls are not enforced to run in any particular order.
// Call order dependency can be enforced by use of InOrder and/or Call.After.
// Call.After can create more varied call order dependencies, but InOrder is
// often more convenient.
//
// The following examples create equivalent call order dependencies.
//
// Example of using Call.After to chain expected call order:
//
//     firstCall := mockObj.EXPECT().SomeMethod(1, "first")
//     secondCall := mockObj.EXPECT().SomeMethod(2, "second").After(firstCall)
//     mockObj.EXPECT().SomeMethod(3, "third").After(secondCall)
//
// Example of using InOrder to declare expected call order:
//
//     gomock.InOrder(
//         mockObj.EXPECT().SomeMethod(1, "first"),
//         mockObj.EXPECT().SomeMethod(2, "second"),
//         mockObj.EXPECT().SomeMethod(3, "third"),
//     )
//
// TODO:
//	- Handle different argument/return types (e.g. ..., chan, map, interface).
package gomock

import "sync"

// A TestReporter is something that can be used to report test failures.
// It is satisfied by the standard library's *testing.T.
type TestReporter interface {
	Errorf(format string, args ...interface{})
	Fatalf(format string, args ...interface{})
}

// A Controller represents the top-level control of a mock ecosystem.
// It defines the scope and lifetime of mock objects, as well as their expectations.
// It is safe to call Controller's methods from multiple goroutines.
type Controller struct {
	mu            sync.Mutex
	t             TestReporter
	expectedCalls callSet
}

func NewController(t TestReporter) *Controller {
	return &Controller{
		t:             t,
		expectedCalls: make(callSet),
	}
}

func (ctrl *Controller) RecordCall(receiver interface{}, method string, args ...interface{}) *Call {
	// TODO: check arity, types.
	margs := make([]Matcher, len(args))
	for i, arg := range args {
		if m, ok := arg.(Matcher); ok {
			margs[i] = m
		} else if arg == nil {
			// Handle nil specially so that passing a nil interface value
			// will match the typed nils of concrete args.
			margs[i] = Nil()
		} else {
			margs[i] = Eq(arg)
		}
	}

	ctrl.mu.Lock()
	defer ctrl.mu.Unlock()

	call := &Call{t: ctrl.t, receiver: receiver, method: method, args: margs, minCalls: 1, maxCalls: 1}

	ctrl.expectedCalls.Add(call)
	return call
}

func (ctrl *Controller) Call(receiver interface{}, method string, args ...interface{}) []interface{} {
	ctrl.mu.Lock()
	defer ctrl.mu.Unlock()

	expected := ctrl.expectedCalls.FindMatch(receiver, method, args)
	if expected == nil {
		ctrl.t.Fatalf("no matching expected call: %T.%v(%v)", receiver, method, args)
	}

	// Two things happen here:
	// * the matching call no longer needs to check prerequite calls,
	// * and the prerequite calls are no longer expected, so remove them.
	preReqCalls := expected.dropPrereqs()
	for _, preReqCall := range preReqCalls {
		ctrl.expectedCalls.Remove(preReqCall)
	}

	rets, action := expected.call(args)
	if expected.exhausted() {
		ctrl.expectedCalls.Remove(expected)
	}

	// Don't hold the lock while doing the call's action (if any)
	// so that actions may execute concurrently.
	// We use the deferred Unlock to capture any panics that happen above;
	// here we add a deferred Lock to balance it.
	ctrl.mu.Unlock()
	defer ctrl.mu.Lock()
	if action != nil {
		action()
	}

	return rets
}

func (ctrl *Controller) Finish() {
	ctrl.mu.Lock()
	defer ctrl.mu.Unlock()

	// If we're currently panicking, probably because this is a deferred call,
	// pass through the panic.
	if err := recover(); err != nil {
		panic(err)
	}

	// Check that all remaining expected calls are satisfied.
	failures := false
	for _, methodMap := range ctrl.expectedCalls {
		for _, calls := range methodMap {
			for _, call := range calls {
				if !call.satisfied() {
					ctrl.t.Errorf("missing call(s) to %v", call)
					failures = true
				}
			}
		}
	}
	if failures {
		ctrl.t.Fatalf("aborting test due to missing call(s)")
	}
}
