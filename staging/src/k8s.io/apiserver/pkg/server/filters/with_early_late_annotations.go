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
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"strings"
	"time"

	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
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
	// invoked before the event is signaled (zero, false) will be returned.
	SignaledAt() (time.Time, bool)
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
		if strings.HasPrefix(path, probe) {
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
func WithShutdownLateAnnotation(handler http.Handler, shutdownInitiated lifecycleEvent, delayDuration time.Duration) http.Handler {
	return withShutdownLateAnnotation(handler, shutdownInitiated, delayDuration, exemptIfHealthProbe, clockutils.RealClock{})
}

func withShutdownLateAnnotation(handler http.Handler, shutdownInitiated lifecycleEvent, delayDuration time.Duration, shouldExemptFn shouldExemptFunc, clock clockutils.PassiveClock) http.Handler {
	if shutdownInitiated == nil || clock == nil || delayDuration == time.Duration(0) || shouldExemptFn == nil {
		klog.Warning("WithShutdownLateAnnotation filter is not being added to the handler chain")
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		select {
		case <-shutdownInitiated.Signaled():
		default:
			handler.ServeHTTP(w, req)
			return
		}

		requestor, exists := request.UserFrom(req.Context())
		if !exists {
			responsewriters.InternalError(w, req, errors.New("no user found for request"))
			return
		}

		defer handler.ServeHTTP(w, req)

		if shouldExemptFn(req) {
			return
		}
		shutdownInitiatedAt, ok := shutdownInitiated.SignaledAt()
		if !ok {
			return
		}
		var isSelf bool
		if requestor.GetName() == user.APIServerUser {
			isSelf = true
		}
		loopback := isLoopback(req.RemoteAddr)

		elapsedSince := clock.Since(shutdownInitiatedAt)
		threshold := int(math.Round((float64(elapsedSince) / float64(delayDuration)) * 100))
		// TODO: 80% is the threshold, if requests arrive after 80% of
		//  shutdown-delay-duration elapses we annotate the request as late=true.
		var isLate bool
		if threshold > 80 {
			isLate = true
		}

		audit.AddAuditAnnotation(req.Context(), "apiserver.k8s.io/shutdown",
			fmt.Sprintf("elapsed=%s threshold=%d%% late=%t self=%t loopback=%t", elapsedSince.Round(time.Second).String(), threshold, isLate, isSelf, loopback))
	})
}

func WithStartupEarlyAnnotation(handler http.Handler, hasBeenReady lifecycleEvent) http.Handler {
	return withStartupEarlyAnnotation(handler, hasBeenReady, exemptIfHealthProbe)
}

func withStartupEarlyAnnotation(handler http.Handler, hasBeenReady lifecycleEvent, shouldExemptFn shouldExemptFunc) http.Handler {
	if hasBeenReady == nil || shouldExemptFn == nil {
		klog.Warning("WithStartupEarlyAnnotation filter is not being added to the handler chain")
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		select {
		case <-hasBeenReady.Signaled():
			handler.ServeHTTP(w, req)
			return
		default:
		}

		requestor, exists := request.UserFrom(req.Context())
		if !exists {
			responsewriters.InternalError(w, req, errors.New("no user found for request"))
			return
		}

		defer handler.ServeHTTP(w, req)

		if requestor.GetName() == user.APIServerUser {
			return
		}

		audit.AddAuditAnnotation(req.Context(), "apiserver.k8s.io/startup", fmt.Sprintf("early=true self=false loopback=%t", isLoopback(req.RemoteAddr)))
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
