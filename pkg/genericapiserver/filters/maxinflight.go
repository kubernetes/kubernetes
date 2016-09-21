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
	"net/http"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/httplog"
)

// Constant for the retry-after interval on rate limiting.
// TODO: maybe make this dynamic? or user-adjustable?
const retryAfter = "1"

// WithMaxInFlightLimit limits the number of in-flight requests to buffer size of the passed in channel.
func WithMaxInFlightLimit(handler http.Handler, limit int, longRunningRequestCheck LongRunningRequestCheck) http.Handler {
	if limit == 0 {
		return handler
	}
	c := make(chan bool, limit)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if longRunningRequestCheck(r) {
			// Skip tracking long running events.
			handler.ServeHTTP(w, r)
			return
		}
		select {
		case c <- true:
			defer func() { <-c }()
			handler.ServeHTTP(w, r)
		default:
			tooManyRequests(r, w)
		}
	})
}

func tooManyRequests(req *http.Request, w http.ResponseWriter) {
	// "Too Many Requests" response is returned before logger is setup for the request.
	// So we need to explicitly log it here.
	defer httplog.NewLogged(req, &w).Log()

	// Return a 429 status indicating "Too Many Requests"
	w.Header().Set("Retry-After", retryAfter)
	http.Error(w, "Too many requests, please try again later.", errors.StatusTooManyRequests)
}
