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

package gomock

import (
	"fmt"
	"reflect"
	"strings"
)

// Call represents an expected call to a mock.
type Call struct {
	t TestReporter // for triggering test failures on invalid call setup

	receiver interface{}   // the receiver of the method call
	method   string        // the name of the method
	args     []Matcher     // the args
	rets     []interface{} // the return values (if any)

	preReqs []*Call // prerequisite calls

	// Expectations
	minCalls, maxCalls int

	numCalls int // actual number made

	// Actions
	doFunc  reflect.Value
	setArgs map[int]reflect.Value
}

// AnyTimes allows the expectation to be called 0 or more times
func (c *Call) AnyTimes() *Call {
	c.minCalls, c.maxCalls = 0, 1e8 // close enough to infinity
	return c
}

// MinTimes requires the call to occur at least n times. If AnyTimes or MaxTimes have not been called, MinTimes also
// sets the maximum number of calls to infinity.
func (c *Call) MinTimes(n int) *Call {
	c.minCalls = n
	if c.maxCalls == 1 {
		c.maxCalls = 1e8
	}
	return c
}

// MaxTimes limits the number of calls to n times. If AnyTimes or MinTimes have not been called, MaxTimes also
// sets the minimum number of calls to 0.
func (c *Call) MaxTimes(n int) *Call {
	c.maxCalls = n
	if c.minCalls == 1 {
		c.minCalls = 0
	}
	return c
}

// Do declares the action to run when the call is matched.
// It takes an interface{} argument to support n-arity functions.
func (c *Call) Do(f interface{}) *Call {
	// TODO: Check arity and types here, rather than dying badly elsewhere.
	c.doFunc = reflect.ValueOf(f)
	return c
}

func (c *Call) Return(rets ...interface{}) *Call {
	mt := c.methodType()
	if len(rets) != mt.NumOut() {
		c.t.Fatalf("wrong number of arguments to Return for %T.%v: got %d, want %d",
			c.receiver, c.method, len(rets), mt.NumOut())
	}
	for i, ret := range rets {
		if got, want := reflect.TypeOf(ret), mt.Out(i); got == want {
			// Identical types; nothing to do.
		} else if got == nil {
			// Nil needs special handling.
			switch want.Kind() {
			case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
				// ok
			default:
				c.t.Fatalf("argument %d to Return for %T.%v is nil, but %v is not nillable",
					i, c.receiver, c.method, want)
			}
		} else if got.AssignableTo(want) {
			// Assignable type relation. Make the assignment now so that the generated code
			// can return the values with a type assertion.
			v := reflect.New(want).Elem()
			v.Set(reflect.ValueOf(ret))
			rets[i] = v.Interface()
		} else {
			c.t.Fatalf("wrong type of argument %d to Return for %T.%v: %v is not assignable to %v",
				i, c.receiver, c.method, got, want)
		}
	}

	c.rets = rets
	return c
}

func (c *Call) Times(n int) *Call {
	c.minCalls, c.maxCalls = n, n
	return c
}

// SetArg declares an action that will set the nth argument's value,
// indirected through a pointer.
func (c *Call) SetArg(n int, value interface{}) *Call {
	if c.setArgs == nil {
		c.setArgs = make(map[int]reflect.Value)
	}
	mt := c.methodType()
	// TODO: This will break on variadic methods.
	// We will need to check those at invocation time.
	if n < 0 || n >= mt.NumIn() {
		c.t.Fatalf("SetArg(%d, ...) called for a method with %d args", n, mt.NumIn())
	}
	// Permit setting argument through an interface.
	// In the interface case, we don't (nay, can't) check the type here.
	at := mt.In(n)
	switch at.Kind() {
	case reflect.Ptr:
		dt := at.Elem()
		if vt := reflect.TypeOf(value); !vt.AssignableTo(dt) {
			c.t.Fatalf("SetArg(%d, ...) argument is a %v, not assignable to %v", n, vt, dt)
		}
	case reflect.Interface:
		// nothing to do
	default:
		c.t.Fatalf("SetArg(%d, ...) referring to argument of non-pointer non-interface type %v", n, at)
	}
	c.setArgs[n] = reflect.ValueOf(value)
	return c
}

// isPreReq returns true if other is a direct or indirect prerequisite to c.
func (c *Call) isPreReq(other *Call) bool {
	for _, preReq := range c.preReqs {
		if other == preReq || preReq.isPreReq(other) {
			return true
		}
	}
	return false
}

// After declares that the call may only match after preReq has been exhausted.
func (c *Call) After(preReq *Call) *Call {
	if preReq.isPreReq(c) {
		msg := fmt.Sprintf(
			"Loop in call order: %v is a prerequisite to %v (possibly indirectly).",
			c, preReq,
		)
		panic(msg)
	}

	c.preReqs = append(c.preReqs, preReq)
	return c
}

// Returns true iff the minimum number of calls have been made.
func (c *Call) satisfied() bool {
	return c.numCalls >= c.minCalls
}

// Returns true iff the maximum number of calls have been made.
func (c *Call) exhausted() bool {
	return c.numCalls >= c.maxCalls
}

func (c *Call) String() string {
	args := make([]string, len(c.args))
	for i, arg := range c.args {
		args[i] = arg.String()
	}
	arguments := strings.Join(args, ", ")
	return fmt.Sprintf("%T.%v(%s)", c.receiver, c.method, arguments)
}

// Tests if the given call matches the expected call.
func (c *Call) matches(args []interface{}) bool {
	if len(args) != len(c.args) {
		return false
	}
	for i, m := range c.args {
		if !m.Matches(args[i]) {
			return false
		}
	}

	// Check that all prerequisite calls have been satisfied.
	for _, preReqCall := range c.preReqs {
		if !preReqCall.satisfied() {
			return false
		}
	}

	return true
}

// dropPrereqs tells the expected Call to not re-check prerequite calls any
// longer, and to return its current set.
func (c *Call) dropPrereqs() (preReqs []*Call) {
	preReqs = c.preReqs
	c.preReqs = nil
	return
}

func (c *Call) call(args []interface{}) (rets []interface{}, action func()) {
	c.numCalls++

	// Actions
	if c.doFunc.IsValid() {
		doArgs := make([]reflect.Value, len(args))
		ft := c.doFunc.Type()
		for i := 0; i < ft.NumIn(); i++ {
			if args[i] != nil {
				doArgs[i] = reflect.ValueOf(args[i])
			} else {
				// Use the zero value for the arg.
				doArgs[i] = reflect.Zero(ft.In(i))
			}
		}
		action = func() { c.doFunc.Call(doArgs) }
	}
	for n, v := range c.setArgs {
		reflect.ValueOf(args[n]).Elem().Set(v)
	}

	rets = c.rets
	if rets == nil {
		// Synthesize the zero value for each of the return args' types.
		mt := c.methodType()
		rets = make([]interface{}, mt.NumOut())
		for i := 0; i < mt.NumOut(); i++ {
			rets[i] = reflect.Zero(mt.Out(i)).Interface()
		}
	}

	return
}

func (c *Call) methodType() reflect.Type {
	recv := reflect.ValueOf(c.receiver)
	for i := 0; i < recv.Type().NumMethod(); i++ {
		if recv.Type().Method(i).Name == c.method {
			return recv.Method(i).Type()
		}
	}
	panic(fmt.Sprintf("gomock: failed finding method %s on %T", c.method, c.receiver))
}

// InOrder declares that the given calls should occur in order.
func InOrder(calls ...*Call) {
	for i := 1; i < len(calls); i++ {
		calls[i].After(calls[i-1])
	}
}
