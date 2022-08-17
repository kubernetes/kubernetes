/*
Copyright 2022 The Kubernetes Authors.

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
	"k8s.io/apiserver/pkg/warning"
	"net/http"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

const (
	// notReadyDebuggerGroup facilitates debugging if the apiserver takes longer
	// to initilize. All request(s) from this designated group will be allowed
	// while the apiserver is being initialized.
	// The apiserver will reject all incoming requests with a 'Retry-After'
	// response header until it has fully initialized, except for
	// requests from this special debugger group.
	notReadyDebuggerGroup = "system:openshift:risky-not-ready-microshift-debugging-group"
)

// WithNotReady rejects any incoming new request(s) with a 'Retry-After'
// response if the specified hasBeenReadyCh channel is still open, with
// the following exceptions:
//   - all request(s) from the designated debugger group is exempt, this
//     helps debug the apiserver if it takes longer to initialize.
//   - local loopback requests (this exempts system:apiserver)
//   - /healthz, /livez, /readyz, /metrics are exempt
//
// It includes new request(s) on a new or an existing TCP connection
// Any new request(s) arriving before hasBeenreadyCh is closed
// are replied with a 503 and the following response headers:
//   - 'Retry-After: N` (so client can retry after N seconds)
func WithNotReady(handler http.Handler, hasBeenReadyCh <-chan struct{}) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		select {
		case <-hasBeenReadyCh:
			handler.ServeHTTP(w, req)
			return
		default:
		}

		requestor, exists := request.UserFrom(req.Context())
		if !exists {
			responsewriters.InternalError(w, req, errors.New("no user found for request"))
			return
		}

		// make sure we exempt:
		//  - local loopback requests (this exempts system:apiserver)
		//  - health probes and metric scraping
		//  - requests from the exempt debugger group.
		if requestor.GetName() == user.APIServerUser ||
			hasExemptPathPrefix(req) ||
			matchesDebuggerGroup(requestor, notReadyDebuggerGroup) {
			warning.AddWarning(req.Context(), "", "The apiserver was still initializing, while this request was being served")
			handler.ServeHTTP(w, req)
			return
		}

		// Return a 503 status asking the client to try again after 5 seconds
		w.Header().Set("Retry-After", "5")
		http.Error(w, "The apiserver hasn't been fully initialized yet, please try again later.",
			http.StatusServiceUnavailable)
	})
}

func matchesDebuggerGroup(requestor user.Info, debugger string) bool {
	for _, group := range requestor.GetGroups() {
		if group == debugger {
			return true
		}
	}
	return false
}
