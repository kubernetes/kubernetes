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

// WithShutdownLateAnnotation, if added to the handler chain, tracks the
// incoming request(s) after the apiserver has initiated the graceful
// shutdown, and annoates the audit event for these request(s) with
// diagnostic information.
// This enables us to identify the actor(s)/load balancer(s) that are sending
// requests to the apiserver late during the server termination.
// It should be placed after (in order of execution) the
// 'WithAuthentication' filter.
func WithShutdownLateAnnotation(handler http.Handler, shutdownInitiated lifecycleEvent, delayDuration time.Duration) http.Handler {
	return withShutdownLateAnnotation(handler, shutdownInitiated, delayDuration, exemptIfHealthProbe, clockutils.RealClock{})
}

// WithStartupEarlyAnnotation annotates the request with an annotation keyed as
// 'apiserver.k8s.io/startup' if the request arrives early (the server is not
// fully initialized yet). It should be placed after (in order of execution)
// the 'WithAuthentication' filter.
func WithStartupEarlyAnnotation(handler http.Handler, hasBeenReady lifecycleEvent) http.Handler {
	return withStartupEarlyAnnotation(handler, hasBeenReady, exemptIfHealthProbe)
}

func withShutdownLateAnnotation(handler http.Handler, shutdownInitiated lifecycleEvent, delayDuration time.Duration, shouldExemptFn shouldExemptFunc, clock clockutils.PassiveClock) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		select {
		case <-shutdownInitiated.Signaled():
		default:
			handler.ServeHTTP(w, req)
			return
		}

		if shouldExemptFn(req) {
			handler.ServeHTTP(w, req)
			return
		}
		shutdownInitiatedAt := shutdownInitiated.SignaledAt()
		if shutdownInitiatedAt == nil {
			handler.ServeHTTP(w, req)
			return
		}

		elapsedSince := clock.Since(*shutdownInitiatedAt)
		// TODO: 80% is the threshold, if requests arrive after 80% of
		//  shutdown-delay-duration elapses we annotate the request as late=true.
		late := lateMsg(delayDuration, elapsedSince, 80)

		// NOTE: some upstream unit tests have authentication disabled and will
		//  fail if we require the requestor to be present in the request
		//  context. Fixing those unit tests will increase the chance of merge
		//  conflict during rebase.
		// This also implies that this filter must be placed after (in order of
		// execution) the 'WithAuthentication' filter.
		self := "self="
		if requestor, exists := request.UserFrom(req.Context()); exists && requestor != nil {
			self = fmt.Sprintf("%s%t", self, requestor.GetName() == user.APIServerUser)
		}

		audit.AddAuditAnnotation(req.Context(), "apiserver.k8s.io/shutdown",
			fmt.Sprintf("%s %s loopback=%t", late, self, isLoopback(req.RemoteAddr)))

		handler.ServeHTTP(w, req)
	})
}

func lateMsg(delayDuration, elapsedSince time.Duration, threshold float64) string {
	if delayDuration == time.Duration(0) {
		return fmt.Sprintf("elapsed=%s threshold= late=%t", elapsedSince.Round(time.Second).String(), true)
	}

	percentElapsed := (float64(elapsedSince) / float64(delayDuration)) * 100
	return fmt.Sprintf("elapsed=%s threshold=%.2f%% late=%t",
		elapsedSince.Round(time.Second).String(), percentElapsed, percentElapsed > threshold)
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
