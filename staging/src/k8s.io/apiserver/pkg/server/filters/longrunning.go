/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"context"
	"net/http"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// BasicLongRunningRequestCheck returns true if the given request has one of the specified verbs or one of the specified subresources, or is a profiler request.
func BasicLongRunningRequestCheck(longRunningVerbs, longRunningSubresources sets.String) apirequest.LongRunningRequestCheck {
	return func(r *http.Request, requestInfo *apirequest.RequestInfo) bool {
		if longRunningVerbs.Has(requestInfo.Verb) {
			return true
		}
		if requestInfo.IsResourceRequest && longRunningSubresources.Has(requestInfo.Subresource) {
			return true
		}
		if !requestInfo.IsResourceRequest && strings.HasPrefix(requestInfo.Path, "/debug/pprof/") {
			return true
		}
		return false
	}
}

// WithLongRunningRequestTermination starts closing long running requests upon receiving TerminationStartCh signal.
//
// It does it by running two additional go routines.
// One for running the request and the second one for intercepting the termination signal and propagating it to the requests.
// If the request is not terminated after receiving a signal from inFlightRequestsFinished it will be forcefully killed.
//
// This filter exists because sometimes propagating the termination signal is not enough.
// It turned out that long running requests might block on:
//  io.Read() for example in https://golang.org/src/net/http/httputil/reverseproxy.go
//  http2.(*serverConn).writeDataFromHandler which might be actually an issue with the std lib itself
//
// Instead of trying to identify current and future issues we provide a filter that ensures terminating long running requests.
//
// Also note that upon receiving termination signal the http server sends
// sends GOAWAY with ErrCodeNo to tell the client we're gracefully shutting down.
// But the connection isn't closed until all current streams are done.
func WithLongRunningRequestTermination(handler http.Handler, longRunningFunc apirequest.LongRunningRequestCheck, terminationStartCh <-chan struct{}, inFlightRequestsFinished <-chan struct{}) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		requestInfo, found := apirequest.RequestInfoFrom(req.Context())
		if !found || longRunningFunc == nil {
			handler.ServeHTTP(w, req)
			return
		}

		isLongRunning := longRunningFunc(req, requestInfo)
		if !isLongRunning {
			handler.ServeHTTP(w, req)
			return
		}

		ctx, cancelCtxFn := context.WithCancel(req.Context())
		defer cancelCtxFn()
		req = req.WithContext(ctx)

		errCh := make(chan interface{}, 2)
		// note that is is okay to leave the errCh open
		// eventually it will be garbage collected
		doneCh := make(chan struct{})

		go func() {
			defer func() {
				err := recover()
				select {
				case errCh <- err:
					return
				}
			}()
			select {
			case <-terminationStartCh:
				cancelCtxFn()
				select {
				case <-inFlightRequestsFinished:
					panic(http.ErrAbortHandler)
				case <-doneCh:
					return
				}
			case <-doneCh:
				return
			}
		}()

		go func() {
			defer func() {
				err := recover()
				select {
				case errCh <- err:
					return
				}
			}()
			handler.ServeHTTP(w, req)
		}()

		err := <-errCh
		close(doneCh)
		if err != nil {
			panic(err)
		}
	})
}
