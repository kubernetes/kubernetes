/*
Copyright 2023 The Kubernetes Authors.

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
	"net"
	"net/http"
	"strings"
	"time"

	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	clockutils "k8s.io/utils/clock"
	netutils "k8s.io/utils/net"
)

type lifecycleEvent interface {
	// Name returns the name of the signal, useful for logging.
	Name() string

	// Signaled returns a channel that is closed when the underlying event
	// has been signaled. Successive calls to Signaled return the same value.
	Signaled() <-chan struct{}

	// SignaledAt returns the time the event was signaled. If SignaledAt is
	// invoked before the event is signaled nil will be returned.
	SignaledAt() *time.Time
}

type shouldExemptFunc func(*http.Request) bool

var (
	// the health probes are not annotated by default
	healthProbes = []string{
		"/readyz",
		"/healthz",
		"/livez",
	}
)

func exemptIfHealthProbe(r *http.Request) bool {
	path := "/" + strings.TrimLeft(r.URL.Path, "/")
	for _, probe := range healthProbes {
		if path == probe {
			return true
		}
	}
	return false
}

// WithShutdownResponseHeader, if added to the handler chain, adds a header
// 'X-OpenShift-Disruption' to the response with the following information:
//
//	shutdown={true|false} shutdown-delay-duration=%s elapsed=%s host=%s
//	 shutdown: whether the server is currently shutting down gracefully.
//	 shutdown-delay-duration: value of --shutdown-delay-duration server run option
//	 elapsed: how much time has elapsed since the server received a TERM signal
//	 host: host name of the server, it is used to identify the server instance
//	       from the others.
//
// This handler will add the response header only if the client opts in by
// adding the 'X-Openshift-If-Disruption' header to the request.
func WithShutdownResponseHeader(handler http.Handler, shutdownInitiated lifecycleEvent, delayDuration time.Duration, apiServerID string) http.Handler {
	return withShutdownResponseHeader(handler, shutdownInitiated, delayDuration, apiServerID, clockutils.RealClock{})
}

// WithStartupEarlyAnnotation annotates the request with an annotation keyed as
// 'apiserver.k8s.io/startup' if the request arrives early (the server is not
// fully initialized yet). It should be placed after (in order of execution)
// the 'WithAuthentication' filter.
func WithStartupEarlyAnnotation(handler http.Handler, hasBeenReady lifecycleEvent) http.Handler {
	return withStartupEarlyAnnotation(handler, hasBeenReady, exemptIfHealthProbe)
}

func withShutdownResponseHeader(handler http.Handler, shutdownInitiated lifecycleEvent, delayDuration time.Duration, apiServerID string, clock clockutils.PassiveClock) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if len(req.Header.Get("X-Openshift-If-Disruption")) == 0 {
			handler.ServeHTTP(w, req)
			return
		}

		msgFn := func(shutdown bool, elapsed time.Duration) string {
			return fmt.Sprintf("shutdown=%t shutdown-delay-duration=%s elapsed=%s host=%s",
				shutdown, delayDuration.Round(time.Second).String(), elapsed.Round(time.Second).String(), apiServerID)
		}

		select {
		case <-shutdownInitiated.Signaled():
		default:
			w.Header().Set("X-OpenShift-Disruption", msgFn(false, time.Duration(0)))
			handler.ServeHTTP(w, req)
			return
		}

		shutdownInitiatedAt := shutdownInitiated.SignaledAt()
		if shutdownInitiatedAt == nil {
			w.Header().Set("X-OpenShift-Disruption", msgFn(true, time.Duration(0)))
			handler.ServeHTTP(w, req)
			return
		}

		w.Header().Set("X-OpenShift-Disruption", msgFn(true, clock.Since(*shutdownInitiatedAt)))
		handler.ServeHTTP(w, req)
	})
}

func withStartupEarlyAnnotation(handler http.Handler, hasBeenReady lifecycleEvent, shouldExemptFn shouldExemptFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		select {
		case <-hasBeenReady.Signaled():
			handler.ServeHTTP(w, req)
			return
		default:
		}

		// NOTE: some upstream unit tests have authentication disabled and will
		//  fail if we require the requestor to be present in the request
		//  context. Fixing those unit tests will increase the chance of merge
		//  conflict during rebase.
		// This also implies that this filter must be placed after (in order of
		// execution) the 'WithAuthentication' filter.
		self := "self="
		if requestor, exists := request.UserFrom(req.Context()); exists && requestor != nil {
			if requestor.GetName() == user.APIServerUser {
				handler.ServeHTTP(w, req)
				return
			}
			self = fmt.Sprintf("%s%t", self, false)
		}

		audit.AddAuditAnnotation(req.Context(), "apiserver.k8s.io/startup",
			fmt.Sprintf("early=true %s loopback=%t", self, isLoopback(req.RemoteAddr)))

		handler.ServeHTTP(w, req)
	})
}

func isLoopback(address string) bool {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		// if the address is missing a port, SplitHostPort will return an error
		// with an empty host, and port value. For such an error, we should
		// continue and try to parse the original address.
		host = address
	}
	if ip := netutils.ParseIPSloppy(host); ip != nil {
		return ip.IsLoopback()
	}

	return false
}
