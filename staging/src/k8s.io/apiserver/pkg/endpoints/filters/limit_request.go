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
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/healthz"
)

// WithLimitUntilHealthy blocks all requests to the handler unless the API is healthy for the first time or it is a health or liveness check
func WithLimitUntilHealthy(handler http.Handler, checks []healthz.HealthChecker) http.Handler {
	hasBeenHealthyBefore := int32(0)
	allowedURLS := map[string]struct{}{
		"healthz": {},
		"livez":   {},
		"readyz":  {},
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		isSelfRequest := false

		urlPaths := strings.Split(r.URL.Path, "/")

		// This checks if the request is a self request.
		if userInfo, ok := genericapirequest.UserFrom(r.Context()); ok {
			if userInfo.GetName() == user.APIServerUser {
				isSelfRequest = true
			}
		}

		// Runs the health checks to ensure that the apiserver is healthy
		storedHasBeenHealthyBefore := atomic.LoadInt32(&hasBeenHealthyBefore)
		if storedHasBeenHealthyBefore == 0 {
			failed := false

			for _, check := range checks {
				if err := check.Check(r); err != nil {
					failed = true
					break
				}
			}

			if !failed {
				atomic.StoreInt32(&hasBeenHealthyBefore, 1)
			}
		}

		if _, ok := allowedURLS[urlPaths[1]]; storedHasBeenHealthyBefore == 1 || isSelfRequest || ok {
			handler.ServeHTTP(w, r)
			return
		}

		responsewriters.InternalError(w, r, fmt.Errorf("API not healthy yet"))
	})
}
