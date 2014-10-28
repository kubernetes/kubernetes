/*
Copyright 2014 Google Inc. All rights reserved.

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

package master

import (
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
)

// handleWhoAmI returns the user-string which this request is authenticated as (if any).
// Useful for debugging authentication.  Always returns HTTP status okay and a human
// readable (not intended as API) description of authentication state of request.
func handleWhoAmI(auth authenticator.Request) func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		if auth == nil {
			w.Write([]byte("NO AUTHENTICATION SUPPORT"))
			return
		}
		userInfo, ok, err := auth.AuthenticateRequest(req)
		if err != nil {
			w.Write([]byte("ERROR WHILE AUTHENTICATING"))
			return
		}
		if !ok {
			w.Write([]byte("NOT AUTHENTICATED"))
			return
		}
		w.Write([]byte("AUTHENTICATED AS " + userInfo.GetName()))
		return
	}
}
