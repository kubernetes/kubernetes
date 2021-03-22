/*
Copyright 2021 The Kubernetes Authors.

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

package finisher

import (
	"context"
	"fmt"
	"net/http"
	goruntime "runtime"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// ResultFunc is a function that returns a rest result and can be run in a goroutine
type ResultFunc func() (runtime.Object, error)

// result stores the return values or panic from a ResultFunc function
type result struct {
	// object stores the response returned by the ResultFunc function
	object runtime.Object
	// err stores the error returned by the ResultFunc function
	err error
	// reason stores the reason from a panic thrown by the ResultFunc function
	reason interface{}
}

// Return processes the result returned by a ResultFunc function
func (r *result) Return() (runtime.Object, error) {
	switch {
	case r.reason != nil:
		// panic has higher precedence, the goroutine executing ResultFunc has panic'd,
		// so propagate a panic to the caller.
		panic(r.reason)
	case r.err != nil:
		return nil, r.err
	default:
		// if we are here, it means neither a panic, nor an error
		if status, ok := r.object.(*metav1.Status); ok {
			// An api.Status object with status != success is considered an "error",
			// which interrupts the normal response flow.
			if status.Status != metav1.StatusSuccess {
				return nil, errors.FromObject(status)
			}
		}
		return r.object, nil
	}
}

// FinishRequest makes a given ResultFunc asynchronous and handles errors returned by the response.
func FinishRequest(ctx context.Context, fn ResultFunc) (runtime.Object, error) {
	// the channel needs to be buffered to prevent the goroutine below from hanging indefinitely
	// when the select statement reads something other than the one the goroutine sends on.
	resultCh := make(chan *result, 1)

	go func() {
		result := &result{}

		// panics don't cross goroutine boundaries, so we have to handle ourselves
		defer func() {
			reason := recover()
			if reason != nil {
				// do not wrap the sentinel ErrAbortHandler panic value
				if reason != http.ErrAbortHandler {
					// Same as stdlib http server code. Manually allocate stack
					// trace buffer size to prevent excessively large logs
					const size = 64 << 10
					buf := make([]byte, size)
					buf = buf[:goruntime.Stack(buf, false)]
					reason = fmt.Sprintf("%v\n%s", reason, buf)
				}

				// store the panic reason into the result.
				result.reason = reason
			}

			// Propagate the result to the parent goroutine
			resultCh <- result
		}()

		if object, err := fn(); err != nil {
			result.err = err
		} else {
			result.object = object
		}
	}()

	select {
	case result := <-resultCh:
		return result.Return()
	case <-ctx.Done():
		return nil, errors.NewTimeoutError(fmt.Sprintf("request did not complete within requested timeout %s", ctx.Err()), 0)
	}
}
