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
	"context"
	"fmt"
	"net/http"
	"strings"

	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// RetryConditionFn is a convenience type used for wrapping a retry condition for WithRetryAfter filter.
type RetryConditionFn func() (bool, func(w http.ResponseWriter), string)

// WithRetryAfter rejects any incoming new request(s) with a 429 if one of the provided condition holds
//
// It includes new request(s) on a new or an existing TCP connection
// Any new request(s) arriving after a condition fulfills
// are replied with a 429 and the following response headers:
//   - 'Retry-After: N` (so client can retry after N seconds, hopefully on a new apiserver instance), where N is defined as [4, 12)
//   -  any optional headers set by a condition function
//
// However, some requests are considered special and always pass to the next handler in the chain.
// As of today, those requests are the ones that:
// - match on the excluded paths
// - originate from the loop back client
func WithRetryAfter(handler http.Handler, conditions []RetryConditionFn, excludedPaths []string, authorizerAttributesFunc func(ctx context.Context) (authorizer.Attributes, error)) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// check if the current path is not explicitly excluded
		for _, excludedPath := range excludedPaths {
			if strings.HasPrefix(req.URL.Path, excludedPath) {
				handler.ServeHTTP(w, req)
				return
			}
		}

		// always allow the loop back client
		ctx := req.Context()
		attribs, err := authorizerAttributesFunc(ctx)
		if err != nil {
			handler.ServeHTTP(w, req)
			return
		}
		if attribs.GetUser().GetName() == user.APIServerUser {
			handler.ServeHTTP(w, req)
			return
		}

		var ok bool
		var rwMutator func(w http.ResponseWriter)
		var reason string
		for _, conditionFn := range conditions {
			ok, rwMutator, reason = conditionFn()
			if ok {
				break
			}
		}

		if !ok {
			handler.ServeHTTP(w, req)
			return
		}

		if rwMutator != nil {
			rwMutator(w)
		}

		// Return a 429 status asking the client to try again after [4, 12) seconds
		retryAfter := rand.Intn(8) + 4
		w.Header().Set("Retry-After", fmt.Sprintf("%d", retryAfter))
		http.Error(w, reason, http.StatusTooManyRequests)
	})
}

// WithRetryOnShutdownDelayCondition meant to be passed to the WithRetryAfter filter
// it rejects any incoming new request(s) with a 429 if the specified channel is closed
// it also sets the following response header: 'Connection: close': to tear down the TCP connection
func WithRetryOnShutdownDelayCondition(ch <-chan struct{}) RetryConditionFn {
	return func() (bool, func(w http.ResponseWriter), string) {
		select {
		case <-ch:
			return true,
				func(rw http.ResponseWriter) {
					// Copied from net/http2 library
					// "Connection" headers aren't allowed in HTTP/2 (RFC 7540, 8.1.2.2),
					// but respect "Connection" == "close" to mean sending a GOAWAY and tearing
					// down the TCP connection when idle, like we do for HTTP/1.
					rw.Header().Set("Connection", "close")
				},
				"The apiserver is shutting down, please try again later"
		default:
			return false, nil, ""
		}
	}
}

// WithRetryWhenHasNotBeenReady meant to be passed to the WithRetryAfter filter
// it rejects any incoming new request(s) with a 429 if the specified channel hasn't been closed
func WithRetryWhenHasNotBeenReady(ch <-chan struct{}) RetryConditionFn {
	return func() (bool, func(w http.ResponseWriter), string) {
		select {
		case <-ch:
			return false, nil, ""
		default:
			return true, nil, "The apiserver hasn't been fully initialized, please try again later"
		}
	}
}

// WithoutRetryOnThePaths holds a list of paths that are excluded from WithRetryAfter filter.
var WithoutRetryOnThePaths = []string{"/readyz", "/livez", "/healthz", "/version", "/logs", "/metrics"}
