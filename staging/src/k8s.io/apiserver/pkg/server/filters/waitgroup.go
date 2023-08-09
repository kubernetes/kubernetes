/*
Copyright 2017 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"net/http"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	utilwaitgroup "k8s.io/apimachinery/pkg/util/waitgroup"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/kubernetes/scheme"
)

// WithWaitGroup adds all non long-running requests to wait group, which is used for graceful shutdown.
func WithWaitGroup(handler http.Handler, longRunning apirequest.LongRunningRequestCheck, wg *utilwaitgroup.SafeWaitGroup) http.Handler {
	// NOTE: both WithWaitGroup and WithRetryAfter must use the same exact isRequestExemptFunc 'isRequestExemptFromRetryAfter,
	// otherwise SafeWaitGroup might wait indefinitely and will prevent the server from shutting down gracefully.
	return withWaitGroup(handler, longRunning, wg, isRequestExemptFromRetryAfter)
}

func withWaitGroup(handler http.Handler, longRunning apirequest.LongRunningRequestCheck, wg *utilwaitgroup.SafeWaitGroup, isRequestExemptFn isRequestExemptFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if !ok {
			// if this happens, the handler chain isn't setup correctly because there is no request info
			responsewriters.InternalError(w, req, errors.New("no RequestInfo found in the context"))
			return
		}

		if longRunning(req, requestInfo) {
			handler.ServeHTTP(w, req)
			return
		}

		if err := wg.Add(1); err != nil {
			// shutdown delay duration has elapsed and SafeWaitGroup.Wait has been invoked,
			// this means 'WithRetryAfter' has started sending Retry-After response.
			// we are going to exempt the same set of requests that WithRetryAfter are
			// exempting from being rejected with a Retry-After response.
			if isRequestExemptFn(req) {
				handler.ServeHTTP(w, req)
				return
			}

			// When apiserver is shutting down, signal clients to retry
			// There is a good chance the client hit a different server, so a tight retry is good for client responsiveness.
			w.Header().Add("Retry-After", "1")
			w.Header().Set("Content-Type", runtime.ContentTypeJSON)
			w.Header().Set("X-Content-Type-Options", "nosniff")
			statusErr := apierrors.NewServiceUnavailable("apiserver is shutting down").Status()
			w.WriteHeader(int(statusErr.Code))
			fmt.Fprintln(w, runtime.EncodeOrDie(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &statusErr))
			return
		}

		defer wg.Done()
		handler.ServeHTTP(w, req)
	})
}
