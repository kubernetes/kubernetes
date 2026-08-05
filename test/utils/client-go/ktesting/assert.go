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

package ktesting

import (
	"context"
	"reflect"
	"slices"

	"github.com/onsi/gomega"
	gtypes "github.com/onsi/gomega/types"

	"k8s.io/kubernetes/test/utils/ktesting"
)

// See [ktesting.FailureError].
type FailureError = ktesting.FailureError

// See [ktesting.ErrFailure].
var ErrFailure error = ktesting.ErrFailure

// assertTestingT implements Fatalf (the only function used by Gomega for
// reporting failures) using TContext.Errorf, i.e. testing continues after a
// failed assertion. The Helper method gets passed through.
type assertTestingT struct {
	TContext
}

var _ gtypes.GomegaTestingT = assertTestingT{}

func (a assertTestingT) Fatalf(format string, args ...any) {
	a.Helper()
	a.Errorf(format, args...)
}

// See [ktesting.Eventually].
func (tCtx TContext) Eventually(arg any) gomega.AsyncAssertion {
	tCtx.Helper()
	return tCtx.newAsyncAssertion(gomega.NewWithT(tCtx).Eventually, arg)
}

// See [ktesting.AssertEventually].
func (tCtx TContext) AssertEventually(arg any) gomega.AsyncAssertion {
	tCtx.Helper()
	return tCtx.newAsyncAssertion(gomega.NewWithT(assertTestingT{tCtx}).Eventually, arg)
}

// See [ktesting.Consistently].
func (tCtx TContext) Consistently(arg any) gomega.AsyncAssertion {
	tCtx.Helper()
	return tCtx.newAsyncAssertion(gomega.NewWithT(tCtx).Consistently, arg)
}

// See [ktesting.AssertConsistently].
func (tCtx TContext) AssertConsistently(arg any) gomega.AsyncAssertion {
	tCtx.Helper()
	return tCtx.newAsyncAssertion(gomega.NewWithT(assertTestingT{tCtx}).Consistently, arg)
}

// newAsyncAssertion must be kept identical to the corresponding newAsyncAssertion in
// the base ktesting, with the addition of the support for TContext from this
// package.
func (tCtx TContext) newAsyncAssertion(eventuallyOrConsistently func(actualOrCtx any, args ...any) gomega.AsyncAssertion, arg any) gomega.AsyncAssertion {
	tCtx.Helper()
	v := reflect.ValueOf(arg)
	if v.Kind() != reflect.Func {
		// Gomega must deal with it.
		return eventuallyOrConsistently(tCtx, arg)
	}
	t := v.Type()
	if t.NumIn() == 0 || !slices.Contains(supportedContextTypes, t.In(0)) {
		// Not a function we can wrap.
		return eventuallyOrConsistently(tCtx, arg)
	}
	// Build a wrapper function with context instead of TContext as first parameter.
	// The wrapper then builds that TContext when called and invokes the actual function.
	in := make([]reflect.Type, t.NumIn())
	in[0] = contextType
	for i := 1; i < t.NumIn(); i++ {
		in[i] = t.In(i)
	}
	out := make([]reflect.Type, t.NumOut())
	for i := range t.NumOut() {
		out[i] = t.Out(i)
	}
	// The last result must always be an error because we need the ability to return assertion
	// failures, so we may have to add an error result value if the function doesn't
	// already have it.
	addErrResult := t.NumOut() == 0 || t.Out(t.NumOut()-1) != errorType
	if addErrResult {
		out = append(out, errorType)
	}
	wrapperType := reflect.FuncOf(in, out, t.IsVariadic())
	wrapper := reflect.MakeFunc(wrapperType, func(args []reflect.Value) (results []reflect.Value) {
		var err error
		var argTCtx any
		var finalize func()
		switch t.In(0) {
		case baseContextType:
			// Construct a base TContext.
			argTCtx, finalize = tCtx.TContext.WithContext(args[0].Interface().(context.Context)).
				WithCancel().
				WithError(&err)
		case k8sContextType:
			// Construct a client-go TContext.
			argTCtx, finalize = tCtx.WithContext(args[0].Interface().(context.Context)).
				WithCancel().
				WithError(&err)
		}
		args[0] = reflect.ValueOf(argTCtx)
		defer func() {
			// This runs *after* finalize.
			// If we are returning normally, then we must inject back the err
			// value that was set by finalize.
			if r := recover(); r != nil {
				// Nope, no results needed.
				panic(r)
			}
			errValue := reflect.ValueOf(err)
			if err == nil {
				// reflect doesn't like this ("returned zero Value").
				// We need a value of the right type.
				errValue = reflect.New(errorType).Elem()
			}
			// If the call panicked and the panic was recoved
			// by finalize(), then results is still nil.
			// We need to fill in null values.
			if len(results) == 0 && t.NumOut() > 0 {
				for t := range t.Outs() {
					results = append(results, reflect.New(t).Elem())
				}
			}
			if addErrResult {
				results = append(results, errValue)
				return
			}
			if results[len(results)-1].IsNil() && err != nil {
				results[len(results)-1] = errValue
			}
		}()
		defer finalize() // Must be called directly, otherwise it cannot recover a panic.
		return v.Call(args)
	})
	return eventuallyOrConsistently(tCtx, wrapper.Interface())
}

var (
	contextType           = reflect.TypeFor[context.Context]()
	errorType             = reflect.TypeFor[error]()
	k8sContextType        = reflect.TypeFor[TContext]()
	baseContextType       = reflect.TypeFor[ktesting.TContext]()
	supportedContextTypes = []reflect.Type{k8sContextType, baseContextType}
)
