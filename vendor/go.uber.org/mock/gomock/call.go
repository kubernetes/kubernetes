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
	"strconv"
	"strings"
)

// Call represents an expected call to a mock.
type Call struct {
	t TestHelper // for triggering test failures on invalid call setup

	receiver   any          // the receiver of the method call
	method     string       // the name of the method
	methodType reflect.Type // the type of the method
	args       []Matcher    // the args
	origin     string       // file and line number of call setup

	preReqs []*Call // prerequisite calls

	// Expectations
	minCalls, maxCalls int

	numCalls int // actual number made

	// actions are called when this Call is called. Each action gets the args and
	// can set the return values by returning a non-nil slice. Actions run in the
	// order they are created.
	actions []func([]any) []any
}

// newCall creates a *Call. It requires the method type in order to support
// unexported methods.
func newCall(t TestHelper, receiver any, method string, methodType reflect.Type, args ...any) *Call {
	t.Helper()

	// TODO: check arity, types.
	mArgs := make([]Matcher, len(args))
	for i, arg := range args {
		if m, ok := arg.(Matcher); ok {
			mArgs[i] = m
		} else if arg == nil {
			// Handle nil specially so that passing a nil interface value
			// will match the typed nils of concrete args.
			mArgs[i] = Nil()
		} else {
			mArgs[i] = Eq(arg)
		}
	}

	// callerInfo's skip should be updated if the number of calls between the user's test
	// and this line changes, i.e. this code is wrapped in another anonymous function.
	// 0 is us, 1 is RecordCallWithMethodType(), 2 is the generated recorder, and 3 is the user's test.
	origin := callerInfo(3)
	actions := []func([]any) []any{func([]any) []any {
		// Synthesize the zero value for each of the return args' types.
		rets := make([]any, methodType.NumOut())
		for i := 0; i < methodType.NumOut(); i++ {
			rets[i] = reflect.Zero(methodType.Out(i)).Interface()
		}
		return rets
	}}
	return &Call{t: t, receiver: receiver, method: method, methodType: methodType,
		args: mArgs, origin: origin, minCalls: 1, maxCalls: 1, actions: actions}
}

// AnyTimes allows the expectation to be called 0 or more times
func (c *Call) AnyTimes() *Call {
	c.minCalls, c.maxCalls = 0, 1e8 // close enough to infinity
	return c
}

// MinTimes requires the call to occur at least n times. If AnyTimes or MaxTimes have not been called or if MaxTimes
// was previously called with 1, MinTimes also sets the maximum number of calls to infinity.
func (c *Call) MinTimes(n int) *Call {
	c.minCalls = n
	if c.maxCalls == 1 {
		c.maxCalls = 1e8
	}
	return c
}

// MaxTimes limits the number of calls to n times. If AnyTimes or MinTimes have not been called or if MinTimes was
// previously called with 1, MaxTimes also sets the minimum number of calls to 0.
func (c *Call) MaxTimes(n int) *Call {
	c.maxCalls = n
	if c.minCalls == 1 {
		c.minCalls = 0
	}
	return c
}

// DoAndReturn declares the action to run when the call is matched.
// The return values from this function are returned by the mocked function.
// It takes an any argument to support n-arity functions.
// The anonymous function must match the function signature mocked method.
func (c *Call) DoAndReturn(f any) *Call {
	// TODO: Check arity and types here, rather than dying badly elsewhere.
	v := reflect.ValueOf(f)

	c.addAction(func(args []any) []any {
		c.t.Helper()
		ft := v.Type()
		if c.methodType.NumIn() != ft.NumIn() {
			if ft.IsVariadic() {
				c.t.Fatalf("wrong number of arguments in DoAndReturn func for %T.%v The function signature must match the mocked method, a variadic function cannot be used.",
					c.receiver, c.method)
			} else {
				c.t.Fatalf("wrong number of arguments in DoAndReturn func for %T.%v: got %d, want %d [%s]",
					c.receiver, c.method, ft.NumIn(), c.methodType.NumIn(), c.origin)
			}
			return nil
		}
		vArgs := make([]reflect.Value, len(args))
		for i := 0; i < len(args); i++ {
			if args[i] != nil {
				vArgs[i] = reflect.ValueOf(args[i])
			} else {
				// Use the zero value for the arg.
				vArgs[i] = reflect.Zero(ft.In(i))
			}
		}
		vRets := v.Call(vArgs)
		rets := make([]any, len(vRets))
		for i, ret := range vRets {
			rets[i] = ret.Interface()
		}
		return rets
	})
	return c
}

// Do declares the action to run when the call is matched. The function's
// return values are ignored to retain backward compatibility. To use the
// return values call DoAndReturn.
// It takes an any argument to support n-arity functions.
// The anonymous function must match the function signature mocked method.
func (c *Call) Do(f any) *Call {
	// TODO: Check arity and types here, rather than dying badly elsewhere.
	v := reflect.ValueOf(f)

	c.addAction(func(args []any) []any {
		c.t.Helper()
		ft := v.Type()
		if c.methodType.NumIn() != ft.NumIn() {
			if ft.IsVariadic() {
				c.t.Fatalf("wrong number of arguments in Do func for %T.%v The function signature must match the mocked method, a variadic function cannot be used.",
					c.receiver, c.method)
			} else {
				c.t.Fatalf("wrong number of arguments in Do func for %T.%v: got %d, want %d [%s]",
					c.receiver, c.method, ft.NumIn(), c.methodType.NumIn(), c.origin)
			}
			return nil
		}
		vArgs := make([]reflect.Value, len(args))
		for i := 0; i < len(args); i++ {
			if args[i] != nil {
				vArgs[i] = reflect.ValueOf(args[i])
			} else {
				// Use the zero value for the arg.
				vArgs[i] = reflect.Zero(ft.In(i))
			}
		}
		v.Call(vArgs)
		return nil
	})
	return c
}

// Return declares the values to be returned by the mocked function call.
func (c *Call) Return(rets ...any) *Call {
	c.t.Helper()

	mt := c.methodType
	if len(rets) != mt.NumOut() {
		c.t.Fatalf("wrong number of arguments to Return for %T.%v: got %d, want %d [%s]",
			c.receiver, c.method, len(rets), mt.NumOut(), c.origin)
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
				c.t.Fatalf("argument %d to Return for %T.%v is nil, but %v is not nillable [%s]",
					i, c.receiver, c.method, want, c.origin)
			}
		} else if got.AssignableTo(want) {
			// Assignable type relation. Make the assignment now so that the generated code
			// can return the values with a type assertion.
			v := reflect.New(want).Elem()
			v.Set(reflect.ValueOf(ret))
			rets[i] = v.Interface()
		} else {
			c.t.Fatalf("wrong type of argument %d to Return for %T.%v: %v is not assignable to %v [%s]",
				i, c.receiver, c.method, got, want, c.origin)
		}
	}

	c.addAction(func([]any) []any {
		return rets
	})

	return c
}

// Times declares the exact number of times a function call is expected to be executed.
func (c *Call) Times(n int) *Call {
	c.minCalls, c.maxCalls = n, n
	return c
}

// SetArg declares an action that will set the nth argument's value,
// indirected through a pointer. Or, in the case of a slice and map, SetArg
// will copy value's elements/key-value pairs into the nth argument.
func (c *Call) SetArg(n int, value any) *Call {
	c.t.Helper()

	mt := c.methodType
	// TODO: This will break on variadic methods.
	// We will need to check those at invocation time.
	if n < 0 || n >= mt.NumIn() {
		c.t.Fatalf("SetArg(%d, ...) called for a method with %d args [%s]",
			n, mt.NumIn(), c.origin)
	}
	// Permit setting argument through an interface.
	// In the interface case, we don't (nay, can't) check the type here.
	at := mt.In(n)
	switch at.Kind() {
	case reflect.Ptr:
		dt := at.Elem()
		if vt := reflect.TypeOf(value); !vt.AssignableTo(dt) {
			c.t.Fatalf("SetArg(%d, ...) argument is a %v, not assignable to %v [%s]",
				n, vt, dt, c.origin)
		}
	case reflect.Interface:
		// nothing to do
	case reflect.Slice:
		// nothing to do
	case reflect.Map:
		// nothing to do
	default:
		c.t.Fatalf("SetArg(%d, ...) referring to argument of non-pointer non-interface non-slice non-map type %v [%s]",
			n, at, c.origin)
	}

	c.addAction(func(args []any) []any {
		v := reflect.ValueOf(value)
		switch reflect.TypeOf(args[n]).Kind() {
		case reflect.Slice:
			setSlice(args[n], v)
		case reflect.Map:
			setMap(args[n], v)
		default:
			reflect.ValueOf(args[n]).Elem().Set(v)
		}
		return nil
	})
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
	c.t.Helper()

	if c == preReq {
		c.t.Fatalf("A call isn't allowed to be its own prerequisite")
	}
	if preReq.isPreReq(c) {
		c.t.Fatalf("Loop in call order: %v is a prerequisite to %v (possibly indirectly).", c, preReq)
	}

	c.preReqs = append(c.preReqs, preReq)
	return c
}

// Returns true if the minimum number of calls have been made.
func (c *Call) satisfied() bool {
	return c.numCalls >= c.minCalls
}

// Returns true if the maximum number of calls have been made.
func (c *Call) exhausted() bool {
	return c.numCalls >= c.maxCalls
}

func (c *Call) String() string {
	args := make([]string, len(c.args))
	for i, arg := range c.args {
		args[i] = arg.String()
	}
	arguments := strings.Join(args, ", ")
	return fmt.Sprintf("%T.%v(%s) %s", c.receiver, c.method, arguments, c.origin)
}

// Tests if the given call matches the expected call.
// If yes, returns nil. If no, returns error with message explaining why it does not match.
func (c *Call) matches(args []any) error {
	if !c.methodType.IsVariadic() {
		if len(args) != len(c.args) {
			return fmt.Errorf("expected call at %s has the wrong number of arguments. Got: %d, want: %d",
				c.origin, len(args), len(c.args))
		}

		for i, m := range c.args {
			if !m.Matches(args[i]) {
				return fmt.Errorf(
					"expected call at %s doesn't match the argument at index %d.\nGot: %v\nWant: %v",
					c.origin, i, formatGottenArg(m, args[i]), m,
				)
			}
		}
	} else {
		if len(c.args) < c.methodType.NumIn()-1 {
			return fmt.Errorf("expected call at %s has the wrong number of matchers. Got: %d, want: %d",
				c.origin, len(c.args), c.methodType.NumIn()-1)
		}
		if len(c.args) != c.methodType.NumIn() && len(args) != len(c.args) {
			return fmt.Errorf("expected call at %s has the wrong number of arguments. Got: %d, want: %d",
				c.origin, len(args), len(c.args))
		}
		if len(args) < len(c.args)-1 {
			return fmt.Errorf("expected call at %s has the wrong number of arguments. Got: %d, want: greater than or equal to %d",
				c.origin, len(args), len(c.args)-1)
		}

		for i, m := range c.args {
			if i < c.methodType.NumIn()-1 {
				// Non-variadic args
				if !m.Matches(args[i]) {
					return fmt.Errorf("expected call at %s doesn't match the argument at index %s.\nGot: %v\nWant: %v",
						c.origin, strconv.Itoa(i), formatGottenArg(m, args[i]), m)
				}
				continue
			}
			// The last arg has a possibility of a variadic argument, so let it branch

			// sample: Foo(a int, b int, c ...int)
			if i < len(c.args) && i < len(args) {
				if m.Matches(args[i]) {
					// Got Foo(a, b, c) want Foo(matcherA, matcherB, gomock.Any())
					// Got Foo(a, b, c) want Foo(matcherA, matcherB, someSliceMatcher)
					// Got Foo(a, b, c) want Foo(matcherA, matcherB, matcherC)
					// Got Foo(a, b) want Foo(matcherA, matcherB)
					// Got Foo(a, b, c, d) want Foo(matcherA, matcherB, matcherC, matcherD)
					continue
				}
			}

			// The number of actual args don't match the number of matchers,
			// or the last matcher is a slice and the last arg is not.
			// If this function still matches it is because the last matcher
			// matches all the remaining arguments or the lack of any.
			// Convert the remaining arguments, if any, into a slice of the
			// expected type.
			vArgsType := c.methodType.In(c.methodType.NumIn() - 1)
			vArgs := reflect.MakeSlice(vArgsType, 0, len(args)-i)
			for _, arg := range args[i:] {
				vArgs = reflect.Append(vArgs, reflect.ValueOf(arg))
			}
			if m.Matches(vArgs.Interface()) {
				// Got Foo(a, b, c, d, e) want Foo(matcherA, matcherB, gomock.Any())
				// Got Foo(a, b, c, d, e) want Foo(matcherA, matcherB, someSliceMatcher)
				// Got Foo(a, b) want Foo(matcherA, matcherB, gomock.Any())
				// Got Foo(a, b) want Foo(matcherA, matcherB, someEmptySliceMatcher)
				break
			}
			// Wrong number of matchers or not match. Fail.
			// Got Foo(a, b) want Foo(matcherA, matcherB, matcherC, matcherD)
			// Got Foo(a, b, c) want Foo(matcherA, matcherB, matcherC, matcherD)
			// Got Foo(a, b, c, d) want Foo(matcherA, matcherB, matcherC, matcherD, matcherE)
			// Got Foo(a, b, c, d, e) want Foo(matcherA, matcherB, matcherC, matcherD)
			// Got Foo(a, b, c) want Foo(matcherA, matcherB)

			return fmt.Errorf("expected call at %s doesn't match the argument at index %s.\nGot: %v\nWant: %v",
				c.origin, strconv.Itoa(i), formatGottenArg(m, args[i:]), c.args[i])
		}
	}

	// Check that all prerequisite calls have been satisfied.
	for _, preReqCall := range c.preReqs {
		if !preReqCall.satisfied() {
			return fmt.Errorf("expected call at %s doesn't have a prerequisite call satisfied:\n%v\nshould be called before:\n%v",
				c.origin, preReqCall, c)
		}
	}

	// Check that the call is not exhausted.
	if c.exhausted() {
		return fmt.Errorf("expected call at %s has already been called the max number of times", c.origin)
	}

	return nil
}

// dropPrereqs tells the expected Call to not re-check prerequisite calls any
// longer, and to return its current set.
func (c *Call) dropPrereqs() (preReqs []*Call) {
	preReqs = c.preReqs
	c.preReqs = nil
	return
}

func (c *Call) call() []func([]any) []any {
	c.numCalls++
	return c.actions
}

// InOrder declares that the given calls should occur in order.
// It panics if the type of any of the arguments isn't *Call or a generated
// mock with an embedded *Call.
func InOrder(args ...any) {
	calls := make([]*Call, 0, len(args))
	for i := 0; i < len(args); i++ {
		if call := getCall(args[i]); call != nil {
			calls = append(calls, call)
			continue
		}
		panic(fmt.Sprintf(
			"invalid argument at position %d of type %T, InOrder expects *gomock.Call or generated mock types with an embedded *gomock.Call",
			i,
			args[i],
		))
	}
	for i := 1; i < len(calls); i++ {
		calls[i].After(calls[i-1])
	}
}

// getCall checks if the parameter is a *Call or a generated struct
// that wraps a *Call and returns the *Call pointer - if neither, it returns nil.
func getCall(arg any) *Call {
	if call, ok := arg.(*Call); ok {
		return call
	}
	t := reflect.ValueOf(arg)
	if t.Kind() != reflect.Ptr && t.Kind() != reflect.Interface {
		return nil
	}
	t = t.Elem()
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.CanInterface() {
			continue
		}
		if call, ok := f.Interface().(*Call); ok {
			return call
		}
	}
	return nil
}

func setSlice(arg any, v reflect.Value) {
	va := reflect.ValueOf(arg)
	for i := 0; i < v.Len(); i++ {
		va.Index(i).Set(v.Index(i))
	}
}

func setMap(arg any, v reflect.Value) {
	va := reflect.ValueOf(arg)
	for _, e := range va.MapKeys() {
		va.SetMapIndex(e, reflect.Value{})
	}
	for _, e := range v.MapKeys() {
		va.SetMapIndex(e, v.MapIndex(e))
	}
}

func (c *Call) addAction(action func([]any) []any) {
	c.actions = append(c.actions, action)
}

func formatGottenArg(m Matcher, arg any) string {
	got := fmt.Sprintf("%v (%T)", arg, arg)
	if gs, ok := m.(GotFormatter); ok {
		got = gs.Got(arg)
	}
	return got
}
