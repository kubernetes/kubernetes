/*
Copyright 2024 The Kubernetes Authors.

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
	"net/http"

	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// WithRequestTimeoutDelegator adds the appropriate request timeout handler to the
// chain, depending on whether PerHandlerReadWriteTimeout is enabled/disabled
// - timeoutHandlerFn: function that creates the legacy timeout handler
// - perRequestHandlerFn: function that creates the per-request deadline handler
// - chain: the chain of handlers constructed so far
func WithRequestTimeoutDelegator(chain http.Handler, timeoutHandlerFn, perRequestHandlerFn func(http.Handler) http.Handler) http.Handler {
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.PerHandlerReadWriteTimeout) {
		// PerHandlerReadWriteTimeout is disabled:
		// return the timeout handler, in keeping with the legacy behavior.
		return timeoutHandlerFn(chain)
	}

	// PerHandlerReadWriteTimeout is enabled: we have to handle
	// http/1x differently than http/2.0
	//
	// HTTP/1x: we want to retain the legacy timeout filter
	//    chain --> per-request --> timeout  --> ...
	//
	//  timeout.ServeHTTP:
	//     |
	//     |-- per-request.ServeHTTP
	//     |     |
	//           |-- chain.ServeHTTP
	//           |
	//
	// this will help us avoid client potentially hanging indefinitely
	// see https://github.com/golang/go/issues/65526
	http1 := perRequestHandlerFn(chain)
	http1 = timeoutHandlerFn(http1)

	// HTTP/2.0: we don't need the legacy timeout filter
	//
	//  per-request.ServeHTTP:
	//     |
	//     |-- chain.ServeHTTP
	//     |
	//
	http2 := perRequestHandlerFn(chain)

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// PerHandlerReadWriteTimeout excludes http/1x requests, so the
		// timeout filter applies to http/1x requests only.
		if req.ProtoMajor == 1 {
			http1.ServeHTTP(w, req)
			return
		}

		http2.ServeHTTP(w, req)
	})
}

// WithPerRequestDeadline applies per-request read/write deadline to the given
// request handler. If the context associated with the request has a valid
// deadline, it is used to set both read and write deadline for the request.
// If the context does not have any deadline, no per-request
// read or write deadline is applied.
func WithPerRequestDeadline(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		deadline, ok := req.Context().Deadline()
		if !ok {
			handler.ServeHTTP(w, req)
			return
		}

		// per-request read and write deadline are set to
		// the overall request timeout.
		ctrl := http.NewResponseController(w)
		if err := ctrl.SetWriteDeadline(deadline); err != nil {
			handleError(w, req, http.StatusInternalServerError, err, "failed to set write deadline for the request")
			return
		}
		if err := ctrl.SetReadDeadline(deadline); err != nil {
			handleError(w, req, http.StatusInternalServerError, err, "failed to set read deadline for the request")
			return
		}

		handler.ServeHTTP(w, req)
	})
}
