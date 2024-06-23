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
)

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
