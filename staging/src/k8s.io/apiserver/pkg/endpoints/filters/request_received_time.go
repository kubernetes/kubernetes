/*
Copyright 2020 The Kubernetes Authors.

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

	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/utils/clock"
)

// WithRequestReceivedTimestamp attaches the ReceivedTimestamp (the time the request reached
// the apiserver) to the context.
func WithRequestReceivedTimestamp(handler http.Handler) http.Handler {
	return withRequestReceivedTimestampWithClock(handler, clock.RealClock{})
}

// The clock is passed as a parameter, handy for unit testing.
func withRequestReceivedTimestampWithClock(handler http.Handler, clock clock.PassiveClock) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		req = req.WithContext(request.WithReceivedTimestamp(ctx, clock.Now()))

		handler.ServeHTTP(w, req)
	})
}
