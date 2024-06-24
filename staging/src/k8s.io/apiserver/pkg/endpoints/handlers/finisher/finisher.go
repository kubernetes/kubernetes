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
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
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

// PostTimeoutLoggerFunc is a function that can be used to log the result returned
// by a ResultFunc after the request had timed out.
// timedOutAt is the time the request had been timed out.
// r is the result returned by the child goroutine.
type PostTimeoutLoggerFunc func(timedOutAt time.Time, r *result)

const (
	// how much time the post-timeout receiver goroutine will wait for the sender
	// (child goroutine executing ResultFunc) to send a result after the request.
	// had timed out.
	postTimeoutLoggerWait = 5 * time.Minute
)

// FinishRequest makes a given ResultFunc asynchronous and handles errors returned by the response.
func FinishRequest(ctx context.Context, fn ResultFunc) (runtime.Object, error) {
	if utilfeature.DefaultFeatureGate.Enabled(features.PerHandlerReadWriteTimeout) {
		return serialFinisher(ctx, fn)
	}
	return asyncFinisher(ctx, fn, postTimeoutLoggerWait, logPostTimeoutResult)
}

// serialFinisher executes the given function in the same goroutine as the caller
func serialFinisher(ctx context.Context, fn ResultFunc) (runtime.Object, error) {
	result := &result{}
	func() {
		// capture the panic here to be rethrown later, this is in
		// keepting with the behavior of the asynchronous finisher
		defer func() {
			if reason := recover(); reason != nil {
				// store the panic reason into the result.
				result.reason = capture(reason)
			}
		}()
		result.object, result.err = fn()
	}()

	return result.Return()
}

// asyncFinisher invokes the given function on a new goroutine (callee), the
// caller goroutine blocks by waiting on a receiving channel for the result.
// the caller will abandon its wait as soon as the given context is
// canceled or expires, and will return a timeout error to the client.
// the callee goroutine may continue to run after the given context expires.
func asyncFinisher(ctx context.Context, fn ResultFunc, postTimeoutWait time.Duration, postTimeoutLogger PostTimeoutLoggerFunc) (runtime.Object, error) {
	// the channel needs to be buffered since the post-timeout receiver goroutine
	// waits up to 5 minutes for the child goroutine to return.
	resultCh := make(chan *result, 1)

	go func() {
		result := &result{}

		// panics don't cross goroutine boundaries, so we have to handle ourselves
		defer func() {
			if reason := recover(); reason != nil {
				// store the panic reason into the result.
				result.reason = capture(reason)
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
		// we are going to send a timeout response to the caller, but the asynchronous goroutine
		// (sender) is still executing the ResultFunc function.
		// kick off a goroutine (receiver) here to wait for the sender (goroutine executing ResultFunc)
		// to send the result and then log details of the result.
		defer func() {
			go func() {
				timedOutAt := time.Now()

				var result *result
				select {
				case result = <-resultCh:
				case <-time.After(postTimeoutWait):
					// we will not wait forever, if we are here then we know that some sender
					// goroutines are taking longer than postTimeoutWait.
				}
				postTimeoutLogger(timedOutAt, result)
			}()
		}()
		return nil, errors.NewTimeoutError(fmt.Sprintf("request did not complete within requested timeout - %s", ctx.Err()), 0)
	}
}

// logPostTimeoutResult logs a panic or an error from the result that the sender (goroutine that is
// executing the ResultFunc function) has sent to the receiver after the request had timed out.
// timedOutAt is the time the request had been timed out
func logPostTimeoutResult(timedOutAt time.Time, r *result) {
	if r == nil {
		// we are using r == nil to indicate that the child goroutine never returned a result.
		metrics.RecordRequestPostTimeout(metrics.PostTimeoutSourceRestHandler, metrics.PostTimeoutHandlerPending)
		klog.Errorf("FinishRequest: post-timeout activity, waited for %s, child goroutine has not returned yet", time.Since(timedOutAt))
		return
	}

	var status string
	switch {
	case r.reason != nil:
		// a non empty reason inside a result object indicates that there was a panic.
		status = metrics.PostTimeoutHandlerPanic
	case r.err != nil:
		status = metrics.PostTimeoutHandlerError
	default:
		status = metrics.PostTimeoutHandlerOK
	}

	metrics.RecordRequestPostTimeout(metrics.PostTimeoutSourceRestHandler, status)
	err := fmt.Errorf("FinishRequest: post-timeout activity - time-elapsed: %s, panicked: %t, err: %v, panic-reason: %v",
		time.Since(timedOutAt), r.reason != nil, r.err, r.reason)
	utilruntime.HandleError(err)
}

func capture(recovered interface{}) interface{} {
	// do not wrap the sentinel ErrAbortHandler panic value
	if recovered == http.ErrAbortHandler {
		return recovered
	}
	// Same as stdlib http server code. Manually allocate stack
	// trace buffer size to prevent excessively large logs
	const size = 64 << 10
	buf := make([]byte, size)
	buf = buf[:goruntime.Stack(buf, false)]
	return fmt.Sprintf("%v\n%s", recovered, buf)
}
