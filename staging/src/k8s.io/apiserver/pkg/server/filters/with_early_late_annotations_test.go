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
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	authenticationuser "k8s.io/apiserver/pkg/authentication/user"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilsclock "k8s.io/utils/clock"
	clocktesting "k8s.io/utils/clock/testing"
)

func TestWithShutdownLateAnnotation(t *testing.T) {
	var (
		shutdownDelayDuration     = 100 * time.Second
		signaledAt                = time.Now()
		elapsedAtWithingThreshold = signaledAt.Add(shutdownDelayDuration - 21*time.Second)
		elapsedAtBeyondThreshold  = signaledAt.Add(shutdownDelayDuration - 19*time.Second)
	)

	tests := []struct {
		name                    string
		shutdownInitiated       func() lifecycleEvent
		delayDuration           time.Duration
		user                    authenticationuser.Info
		clock                   func() utilsclock.PassiveClock
		url                     string
		remoteAddr              string
		handlerInvoked          int
		statusCodeExpected      int
		annotationShouldContain string
	}{
		{
			name: "shutdown is not initiated",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: make(chan struct{})}
			},
			handlerInvoked:     1,
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "shutdown initiated, health probes are not annotated",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel()}
			},
			url:                "/readyz?verbos=1",
			user:               &authenticationuser.DefaultInfo{Name: authenticationuser.Anonymous},
			handlerInvoked:     1,
			statusCodeExpected: http.StatusOK,
		},
		// use cases where the request will be annotated
		{
			name: "shutdown initiated, no user in request context",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel(), at: &signaledAt}
			},
			handlerInvoked:          1,
			statusCodeExpected:      http.StatusOK,
			annotationShouldContain: "self= loopback=",
		},
		{
			name: "shutdown initiated, self=true",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel(), at: &signaledAt}
			},
			user:                    &authenticationuser.DefaultInfo{Name: authenticationuser.APIServerUser},
			handlerInvoked:          1,
			statusCodeExpected:      http.StatusOK,
			annotationShouldContain: "self=true",
		},
		{
			name: "shutdown initiated, self=false",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel(), at: &signaledAt}
			},
			user:                    &authenticationuser.DefaultInfo{Name: authenticationuser.Anonymous},
			handlerInvoked:          1,
			statusCodeExpected:      http.StatusOK,
			annotationShouldContain: "self=false",
		},
		{
			name: "shutdown initiated, loopback=true",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel(), at: &signaledAt}
			},
			user:                    &authenticationuser.DefaultInfo{Name: authenticationuser.Anonymous},
			remoteAddr:              "127.0.0.1:80",
			handlerInvoked:          1,
			statusCodeExpected:      http.StatusOK,
			annotationShouldContain: "loopback=true",
		},
		{
			name: "shutdown initiated, loopback=false",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel(), at: &signaledAt}
			},
			user:                    &authenticationuser.DefaultInfo{Name: authenticationuser.Anonymous},
			remoteAddr:              "www.foo.bar:80",
			handlerInvoked:          1,
			statusCodeExpected:      http.StatusOK,
			annotationShouldContain: "loopback=false",
		},
		{
			name: "shutdown initiated, shutdown delay duration is zero",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel(), at: &signaledAt}
			},
			delayDuration: time.Duration(0),
			clock: func() utilsclock.PassiveClock {
				return clocktesting.NewFakeClock(elapsedAtWithingThreshold)
			},
			user:                    &authenticationuser.DefaultInfo{Name: authenticationuser.Anonymous},
			handlerInvoked:          1,
			statusCodeExpected:      http.StatusOK,
			annotationShouldContain: "elapsed=1m19s threshold= late=true",
		},
		{
			name: "shutdown initiated, within 80%",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel(), at: &signaledAt}
			},
			delayDuration: shutdownDelayDuration,
			clock: func() utilsclock.PassiveClock {
				return clocktesting.NewFakeClock(elapsedAtWithingThreshold)
			},
			user:                    &authenticationuser.DefaultInfo{Name: authenticationuser.Anonymous},
			handlerInvoked:          1,
			statusCodeExpected:      http.StatusOK,
			annotationShouldContain: "elapsed=1m19s threshold=79.00% late=false self=false loopback=false",
		},
		{
			name: "shutdown initiated, outside 80%",
			shutdownInitiated: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel(), at: &signaledAt}
			},
			delayDuration: shutdownDelayDuration,
			clock: func() utilsclock.PassiveClock {
				return clocktesting.NewFakeClock(elapsedAtBeyondThreshold)
			},
			user:                    &authenticationuser.DefaultInfo{Name: authenticationuser.Anonymous},
			handlerInvoked:          1,
			statusCodeExpected:      http.StatusOK,
			annotationShouldContain: "elapsed=1m21s threshold=81.00% late=true self=false loopback=false",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var handlerInvoked int
			handler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				handlerInvoked++
				w.WriteHeader(http.StatusOK)
			})

			event := test.shutdownInitiated()
			var clock utilsclock.PassiveClock = utilsclock.RealClock{}
			if test.clock != nil {
				clock = test.clock()
			}
			target := withShutdownLateAnnotation(handler, event, test.delayDuration, exemptIfHealthProbe, clock)

			url := "/api/v1/namespaces"
			if test.url != "" {
				url = test.url
			}
			req, err := http.NewRequest(http.MethodGet, url, nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}
			if test.remoteAddr != "" {
				req.RemoteAddr = test.remoteAddr
			}

			ctx := req.Context()
			if test.user != nil {
				ctx = apirequest.WithUser(ctx, test.user)
			}
			ctx = audit.WithAuditContext(ctx)
			req = req.WithContext(ctx)

			ac := audit.AuditContextFrom(req.Context())
			if ac == nil {
				t.Fatalf("expected audit context inside the request context")
			}
			ac.Event = &auditinternal.Event{
				Level: auditinternal.LevelMetadata,
			}

			w := httptest.NewRecorder()
			w.Code = 0
			target.ServeHTTP(w, req)

			if test.handlerInvoked != handlerInvoked {
				t.Errorf("expected the handler to be invoked: %d timed, but got: %d", test.handlerInvoked, handlerInvoked)
			}
			if test.statusCodeExpected != w.Result().StatusCode {
				t.Errorf("expected status code: %d, but got: %d", test.statusCodeExpected, w.Result().StatusCode)
			}

			key := "apiserver.k8s.io/shutdown"
			switch {
			case len(test.annotationShouldContain) == 0:
				if valueGot, ok := ac.Event.Annotations[key]; ok {
					t.Errorf("did not expect annotation to be added, but got: %s", valueGot)
				}
			default:
				if valueGot, ok := ac.Event.Annotations[key]; !ok || !strings.Contains(valueGot, test.annotationShouldContain) {
					t.Logf("got: %s", valueGot)
					t.Errorf("expected annotation to match, diff: %s", cmp.Diff(test.annotationShouldContain, valueGot))
				}
			}
		})
	}
}

func TestWithStartupEarlyAnnotation(t *testing.T) {
	tests := []struct {
		name               string
		readySignalFn      func() lifecycleEvent
		user               authenticationuser.Info
		remoteAddr         string
		handlerInvoked     int
		statusCodeExpected int
		annotationExpected string
	}{
		{
			name: "server is ready",
			readySignalFn: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: newClosedChannel()}
			},
			handlerInvoked:     1,
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "server not ready, no user in request context",
			readySignalFn: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: make(chan struct{})}
			},
			handlerInvoked:     1,
			statusCodeExpected: http.StatusOK,
			annotationExpected: "early=true self= loopback=false",
		},
		{
			name: "server not ready, self is true, not annotated",
			readySignalFn: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: make(chan struct{})}
			},
			user:               &authenticationuser.DefaultInfo{Name: authenticationuser.APIServerUser},
			handlerInvoked:     1,
			statusCodeExpected: http.StatusOK,
		},
		{
			name: "server not ready, self is false, request is annotated",
			readySignalFn: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: make(chan struct{})}
			},
			user:               &authenticationuser.DefaultInfo{Name: authenticationuser.Anonymous},
			handlerInvoked:     1,
			statusCodeExpected: http.StatusOK,
			annotationExpected: "early=true self=false loopback=false",
		},
		{
			name: "server not ready, self is false, looback is true, request is annotated",
			readySignalFn: func() lifecycleEvent {
				return fakeLifecycleSignal{ch: make(chan struct{})}
			},
			user:               &authenticationuser.DefaultInfo{Name: authenticationuser.Anonymous},
			remoteAddr:         "127.0.0.1:8080",
			handlerInvoked:     1,
			statusCodeExpected: http.StatusOK,
			annotationExpected: "early=true self=false loopback=true",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var handlerInvoked int
			handler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				handlerInvoked++
				w.WriteHeader(http.StatusOK)
			})

			event := test.readySignalFn()
			target := WithStartupEarlyAnnotation(handler, event)

			req, err := http.NewRequest(http.MethodGet, "/api/v1/namespaces", nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}
			if test.remoteAddr != "" {
				req.RemoteAddr = test.remoteAddr
			}

			ctx := req.Context()
			if test.user != nil {
				ctx = apirequest.WithUser(ctx, test.user)
			}
			ctx = audit.WithAuditContext(ctx)
			req = req.WithContext(ctx)

			ac := audit.AuditContextFrom(req.Context())
			if ac == nil {
				t.Fatalf("expected audit context inside the request context")
			}
			ac.Event = auditinternal.Event{
				Level: auditinternal.LevelMetadata,
			}

			w := httptest.NewRecorder()
			w.Code = 0
			target.ServeHTTP(w, req)

			if test.handlerInvoked != handlerInvoked {
				t.Errorf("expected the handler to be invoked: %d timed, but got: %d", test.handlerInvoked, handlerInvoked)
			}
			if test.statusCodeExpected != w.Result().StatusCode {
				t.Errorf("expected status code: %d, but got: %d", test.statusCodeExpected, w.Result().StatusCode)
			}

			key := "apiserver.k8s.io/startup"
			switch {
			case len(test.annotationExpected) == 0:
				if valueGot, ok := ac.Event.Annotations[key]; ok {
					t.Errorf("did not expect annotation to be added, but got: %s", valueGot)
				}
			default:
				if valueGot, ok := ac.Event.Annotations[key]; !ok || test.annotationExpected != valueGot {
					t.Errorf("expected annotation: %s, but got: %s", test.annotationExpected, valueGot)
				}
			}
		})
	}
}

func TestIsLoopback(t *testing.T) {
	tests := []struct {
		address string
		want    bool
	}{
		{
			address: "www.foo.bar:80",
			want:    false,
		},
		{
			address: "www.foo.bar",
			want:    false,
		},
		{
			address: "127.0.0.1:8080",
			want:    true,
		},
		{
			address: "127.0.0.1",
			want:    true,
		},
		{
			address: "192.168.0.1",
			want:    false,
		},
		// localhost does not work
		{
			address: "localhost:8080",
			want:    false,
		},
		{
			address: "localhost",
			want:    false,
		},
	}

	for _, test := range tests {
		t.Run(test.address, func(t *testing.T) {
			if got := isLoopback(test.address); test.want != got {
				t.Errorf("expected isLoopback to return: %t, but got: %t", test.want, got)
			}
		})
	}
}

func TestExemptIfHealthProbe(t *testing.T) {
	tests := []struct {
		path   string
		exempt bool
	}{
		{
			path:   "/apis/v1/foo/bar",
			exempt: false,
		},
		{
			path:   "/readyz",
			exempt: true,
		},
		{
			path:   "http://foo.bar///healthz?verbose=1",
			exempt: true,
		},
		{
			path:   "/livez",
			exempt: true,
		},
	}

	for _, test := range tests {
		t.Run(test.path, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, test.path, nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}
			if got := exemptIfHealthProbe(req); test.exempt != got {
				t.Errorf("expected exemptIfHealthProbe to return: %t, but got: %t", test.exempt, got)
			}
		})
	}
}

type fakeLifecycleSignal struct {
	ch <-chan struct{}
	at *time.Time
}

func (s fakeLifecycleSignal) Name() string              { return "initiated" }
func (s fakeLifecycleSignal) Signaled() <-chan struct{} { return s.ch }
func (s fakeLifecycleSignal) SignaledAt() *time.Time    { return s.at }

func newClosedChannel() <-chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}
