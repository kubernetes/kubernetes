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

// WithRetryAfter rejects any incoming new request(s) with a 429
// if the specified shutdownDelayDurationElapsedCh channel is closed
//
// It includes new request(s) on a new or an existing TCP connection
// Any new request(s) arriving after shutdownDelayDurationElapsedCh is closed
// are replied with a 429 and the following response headers:
//   - 'Retry-After: N` (so client can retry after N seconds, hopefully on a new apiserver instance)
//   - 'Connection: close': tear down the TCP connection
//
// TODO: is there a way to merge WithWaitGroup and this filter?
func WithRetryAfter(handler http.Handler, shutdownDelayDurationElapsedCh <-chan struct{}) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		select {
		case <-shutdownDelayDurationElapsedCh:
		default:
			handler.ServeHTTP(w, req)
			return
		}

		// Copied from net/http2 library
		// "Connection" headers aren't allowed in HTTP/2 (RFC 7540, 8.1.2.2),
		// but respect "Connection" == "close" to mean sending a GOAWAY and tearing
		// down the TCP connection when idle, like we do for HTTP/1.
		w.Header().Set("Connection", "close")

		// Return a 429 status asking the cliet to try again after 5 seconds
		w.Header().Set("Retry-After", "5")
		http.Error(w, "The apiserver is shutting down, please try again later.", http.StatusTooManyRequests)
	})
}
