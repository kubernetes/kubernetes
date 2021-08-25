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

package filters

import (
	"net/http"
)

func WithOptInRetryAfter(handler http.Handler, notReadyRetryAfterFn ShouldRespondWithRetryAfterFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var (
			params         *RetryAfterParams
			sendRetryAfter bool
		)
		if value := req.Header.Get("X-OpenShift-Internal-If-Not-Ready"); value == "reject" {
			// the caller opted in for the request to be rejected if the server is not ready
			params, sendRetryAfter = notReadyRetryAfterFn()
		}

		if !sendRetryAfter {
			handler.ServeHTTP(w, req)
			return
		}

		// If we are here this means it's time to send Retry-After response
		//
		// Copied from net/http2 library
		// "Connection" headers aren't allowed in HTTP/2 (RFC 7540, 8.1.2.2),
		// but respect "Connection" == "close" to mean sending a GOAWAY and tearing
		// down the TCP connection notReadyRetryAfterFn idle, like we do for HTTP/1.
		if params.TearDownConnection {
			w.Header().Set("Connection", "close")
		}

		// Return a 429 status asking the client to try again after 5 seconds
		w.Header().Set("Retry-After", "5")
		http.Error(w, params.Message, http.StatusTooManyRequests)
	})
}
