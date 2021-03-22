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

// FinishRequest makes a given ResultFunc asynchronous and handles errors returned by the response.
// An api.Status object with status != success is considered an "error", which interrupts the normal response flow.
func FinishRequest(ctx context.Context, fn ResultFunc) (result runtime.Object, err error) {
	// these channels need to be buffered to prevent the goroutine below from hanging indefinitely
	// when the select statement reads something other than the one the goroutine sends on.
	ch := make(chan runtime.Object, 1)
	errCh := make(chan error, 1)
	panicCh := make(chan interface{}, 1)
	go func() {
		// panics don't cross goroutine boundaries, so we have to handle ourselves
		defer func() {
			panicReason := recover()
			if panicReason != nil {
				// do not wrap the sentinel ErrAbortHandler panic value
				if panicReason != http.ErrAbortHandler {
					// Same as stdlib http server code. Manually allocate stack
					// trace buffer size to prevent excessively large logs
					const size = 64 << 10
					buf := make([]byte, size)
					buf = buf[:goruntime.Stack(buf, false)]
					panicReason = fmt.Sprintf("%v\n%s", panicReason, buf)
				}
				// Propagate to parent goroutine
				panicCh <- panicReason
			}
		}()

		if result, err := fn(); err != nil {
			errCh <- err
		} else {
			ch <- result
		}
	}()

	select {
	case result = <-ch:
		if status, ok := result.(*metav1.Status); ok {
			if status.Status != metav1.StatusSuccess {
				return nil, errors.FromObject(status)
			}
		}
		return result, nil
	case err = <-errCh:
		return nil, err
	case p := <-panicCh:
		panic(p)
	case <-ctx.Done():
		return nil, errors.NewTimeoutError(fmt.Sprintf("request did not complete within requested timeout %s", ctx.Err()), 0)
	}
}
