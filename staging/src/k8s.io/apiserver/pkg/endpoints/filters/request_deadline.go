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
	"context"
	"errors"
	"fmt"
	"k8s.io/klog/v2"
	"net/http"
	"time"

	"k8s.io/apiserver/pkg/endpoints/request"
)

var (
	// The 'timeout' query parameter in the request URL has an invalid timeout specifier
	errInvalidTimeoutInURL = errors.New("invalid timeout specified in the request URL")

	// The timeout specified in the request URL exceeds the global maximum timeout allowed by the apiserver.
	errTimeoutExceedsMaximumAllowed = errors.New("timeout specified in the request URL exceeds the maximum timeout allowed by the server")
)

// WithRequestDeadline determines the deadline of the given request and sets a new context with the appropriate timeout.
// requestTimeoutMaximum specifies the default request timeout value
func WithRequestDeadline(handler http.Handler, longRunning request.LongRunningRequestCheck, requestTimeoutMaximum time.Duration) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()

		requestInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			handleError(w, req, http.StatusInternalServerError, fmt.Errorf("no RequestInfo found in context, handler chain must be wrong"))
			return
		}
		if longRunning(req, requestInfo) {
			handler.ServeHTTP(w, req)
			return
		}

		userSpecifiedTimeout, ok, err := parseTimeout(req)
		if err != nil {
			statusCode := http.StatusInternalServerError
			if err == errInvalidTimeoutInURL {
				statusCode = http.StatusBadRequest
			}

			handleError(w, req, statusCode, err)
			return
		}

		timeout := requestTimeoutMaximum
		if ok {
			if userSpecifiedTimeout > requestTimeoutMaximum {
				handleError(w, req, http.StatusBadRequest, errTimeoutExceedsMaximumAllowed)
				return
			}

			timeout = userSpecifiedTimeout
		}

		ctx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		req = req.WithContext(ctx)
		handler.ServeHTTP(w, req)
	})
}

// parseTimeout parses the given HTTP request URL and extracts the timeout query parameter
// value if specified by the user.
// If a timeout is not specified the function returns false and err is set to nil
// If the value specified is malformed then the function returns false and err is set
func parseTimeout(req *http.Request) (time.Duration, bool, error) {
	value := req.URL.Query().Get("timeout")
	if value == "" {
		return 0, false, nil
	}

	timeout, err := time.ParseDuration(value)
	if err != nil {
		return 0, false, errInvalidTimeoutInURL
	}

	return timeout, true, nil
}

func handleError(w http.ResponseWriter, r *http.Request, code int, err error) {
	errorMsg := fmt.Sprintf("Error - %s: %#v", err.Error(), r.RequestURI)
	http.Error(w, errorMsg, code)
	klog.Errorf(err.Error())
}
