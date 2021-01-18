/*
Copyright 2016 The Kubernetes Authors.

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
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"golang.org/x/net/http2"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
)

type fakeAuditSink struct {
	lock   sync.Mutex
	events []*auditinternal.Event
}

func (s *fakeAuditSink) ProcessEvents(evs ...*auditinternal.Event) bool {
	s.lock.Lock()
	defer s.lock.Unlock()
	for _, e := range evs {
		event := e.DeepCopy()
		s.events = append(s.events, event)
	}
	return true
}

func (s *fakeAuditSink) Events() []*auditinternal.Event {
	s.lock.Lock()
	defer s.lock.Unlock()
	return append([]*auditinternal.Event{}, s.events...)
}

func (s *fakeAuditSink) Pop(timeout time.Duration) (*auditinternal.Event, error) {
	var result *auditinternal.Event
	err := wait.Poll(50*time.Millisecond, wait.ForeverTestTimeout, wait.ConditionFunc(func() (done bool, err error) {
		s.lock.Lock()
		defer s.lock.Unlock()
		if len(s.events) == 0 {
			return false, nil
		}
		result = s.events[0]
		s.events = s.events[1:]
		return true, nil
	}))
	return result, err
}

type simpleResponseWriter struct{}

var _ http.ResponseWriter = &simpleResponseWriter{}

func (*simpleResponseWriter) WriteHeader(code int)         {}
func (*simpleResponseWriter) Write(bs []byte) (int, error) { return len(bs), nil }
func (*simpleResponseWriter) Header() http.Header          { return http.Header{} }

type fancyResponseWriter struct {
	simpleResponseWriter
}

func (*fancyResponseWriter) CloseNotify() <-chan bool { return nil }

func (*fancyResponseWriter) Flush() {}

func (*fancyResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) { return nil, nil, nil }

func TestConstructResponseWriter(t *testing.T) {
	actual := decorateResponseWriter(&simpleResponseWriter{}, nil, nil, nil)
	switch v := actual.(type) {
	case *auditResponseWriter:
	default:
		t.Errorf("Expected auditResponseWriter, got %v", reflect.TypeOf(v))
	}

	actual = decorateResponseWriter(&fancyResponseWriter{}, nil, nil, nil)
	switch v := actual.(type) {
	case *fancyResponseWriterDelegator:
	default:
		t.Errorf("Expected fancyResponseWriterDelegator, got %v", reflect.TypeOf(v))
	}
}

func TestDecorateResponseWriterWithoutChannel(t *testing.T) {
	ev := &auditinternal.Event{}
	actual := decorateResponseWriter(&simpleResponseWriter{}, ev, nil, nil)

	// write status. This will not block because firstEventSentCh is nil
	actual.WriteHeader(42)
	if ev.ResponseStatus == nil {
		t.Fatalf("Expected ResponseStatus to be non-nil")
	}
	if ev.ResponseStatus.Code != 42 {
		t.Errorf("expected status code 42, got %d", ev.ResponseStatus.Code)
	}
}

func TestDecorateResponseWriterWithImplicitWrite(t *testing.T) {
	ev := &auditinternal.Event{}
	actual := decorateResponseWriter(&simpleResponseWriter{}, ev, nil, nil)

	// write status. This will not block because firstEventSentCh is nil
	actual.Write([]byte("foo"))
	if ev.ResponseStatus == nil {
		t.Fatalf("Expected ResponseStatus to be non-nil")
	}
	if ev.ResponseStatus.Code != 200 {
		t.Errorf("expected status code 200, got %d", ev.ResponseStatus.Code)
	}
}

func TestDecorateResponseWriterChannel(t *testing.T) {
	sink := &fakeAuditSink{}
	ev := &auditinternal.Event{}
	actual := decorateResponseWriter(&simpleResponseWriter{}, ev, sink, nil)

	done := make(chan struct{})
	go func() {
		t.Log("Writing status code 42")
		actual.WriteHeader(42)
		t.Log("Finished writing status code 42")
		close(done)

		actual.Write([]byte("foo"))
	}()

	// sleep some time to give write the possibility to do wrong stuff
	time.Sleep(100 * time.Millisecond)

	t.Log("Waiting for event in the channel")
	ev1, err := sink.Pop(time.Second)
	if err != nil {
		t.Fatal("Timeout waiting for events")
	}
	t.Logf("Seen event with status %v", ev1.ResponseStatus)

	if !reflect.DeepEqual(ev, ev1) {
		t.Fatalf("ev1 and ev must be equal")
	}

	<-done
	t.Log("Seen the go routine finished")

	// write again
	_, err = actual.Write([]byte("foo"))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

type fakeHTTPHandler struct{}

func (*fakeHTTPHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(200)
}

func TestAudit(t *testing.T) {
	shortRunningPath := "/api/v1/namespaces/default/pods/foo"
	longRunningPath := "/api/v1/namespaces/default/pods?watch=true"

	delay := 500 * time.Millisecond

	for _, test := range []struct {
		desc       string
		path       string
		verb       string
		auditID    string
		omitStages []auditinternal.Stage
		handler    func(http.ResponseWriter, *http.Request)
		expected   []auditinternal.Event
		respHeader bool
	}{
		// short running requests with read-only verb
		{
			"read-only empty",
			shortRunningPath,
			"GET",
			"",
			nil,
			func(http.ResponseWriter, *http.Request) {},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "get",
					RequestURI: shortRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "get",
					RequestURI:     shortRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
			},
			false,
		},
		{
			"short running with auditID",
			shortRunningPath,
			"GET",
			uuid.New().String(),
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				w.Write([]byte("foo"))
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "get",
					RequestURI: shortRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "get",
					RequestURI:     shortRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
			},
			true,
		},
		{
			"read-only panic",
			shortRunningPath,
			"GET",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				panic("kaboom")
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "get",
					RequestURI: shortRunningPath,
				},
				{
					Stage:          auditinternal.StagePanic,
					Verb:           "get",
					RequestURI:     shortRunningPath,
					ResponseStatus: &metav1.Status{Code: 500},
				},
			},
			false,
		},
		// short running request with non-read-only verb
		{
			"writing empty",
			shortRunningPath,
			"PUT",
			"",
			nil,
			func(http.ResponseWriter, *http.Request) {},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "update",
					RequestURI: shortRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "update",
					RequestURI:     shortRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
			},
			false,
		},
		{
			"writing sleep",
			shortRunningPath,
			"PUT",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				w.Write([]byte("foo"))
				time.Sleep(delay)
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "update",
					RequestURI: shortRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "update",
					RequestURI:     shortRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
			},
			true,
		},
		{
			"writing 403+write",
			shortRunningPath,
			"PUT",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				w.WriteHeader(403)
				w.Write([]byte("foo"))
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "update",
					RequestURI: shortRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "update",
					RequestURI:     shortRunningPath,
					ResponseStatus: &metav1.Status{Code: 403},
				},
			},
			true,
		},
		{
			"writing panic",
			shortRunningPath,
			"PUT",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				panic("kaboom")
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "update",
					RequestURI: shortRunningPath,
				},
				{
					Stage:          auditinternal.StagePanic,
					Verb:           "update",
					RequestURI:     shortRunningPath,
					ResponseStatus: &metav1.Status{Code: 500},
				},
			},
			false,
		},
		{
			"writing write+panic",
			shortRunningPath,
			"PUT",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				w.Write([]byte("foo"))
				panic("kaboom")
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "update",
					RequestURI: shortRunningPath,
				},
				{
					Stage:          auditinternal.StagePanic,
					Verb:           "update",
					RequestURI:     shortRunningPath,
					ResponseStatus: &metav1.Status{Code: 500},
				},
			},
			true,
		},
		// long running requests
		{
			"empty longrunning",
			longRunningPath,
			"GET",
			"",
			nil,
			func(http.ResponseWriter, *http.Request) {},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "watch",
					RequestURI: longRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseStarted,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
			},
			false,
		},
		{
			"empty longrunning with audit id",
			longRunningPath,
			"GET",
			uuid.New().String(),
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				w.Write([]byte("foo"))
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "watch",
					RequestURI: longRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseStarted,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
			},
			true,
		},
		{
			"sleep longrunning",
			longRunningPath,
			"GET",
			"",
			nil,
			func(http.ResponseWriter, *http.Request) {
				time.Sleep(delay)
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "watch",
					RequestURI: longRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseStarted,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
			},
			false,
		},
		{
			"sleep+403 longrunning",
			longRunningPath,
			"GET",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				time.Sleep(delay)
				w.WriteHeader(403)
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "watch",
					RequestURI: longRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseStarted,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 403},
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 403},
				},
			},
			true,
		},
		{
			"write longrunning",
			longRunningPath,
			"GET",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				w.Write([]byte("foo"))
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "watch",
					RequestURI: longRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseStarted,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
			},
			true,
		},
		{
			"403+write longrunning",
			longRunningPath,
			"GET",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				w.WriteHeader(403)
				w.Write([]byte("foo"))
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "watch",
					RequestURI: longRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseStarted,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 403},
				},
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 403},
				},
			},
			true,
		},
		{
			"panic longrunning",
			longRunningPath,
			"GET",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				panic("kaboom")
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "watch",
					RequestURI: longRunningPath,
				},
				{
					Stage:          auditinternal.StagePanic,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 500},
				},
			},
			false,
		},
		{
			"write+panic longrunning",
			longRunningPath,
			"GET",
			"",
			nil,
			func(w http.ResponseWriter, req *http.Request) {
				w.Write([]byte("foo"))
				panic("kaboom")
			},
			[]auditinternal.Event{
				{
					Stage:      auditinternal.StageRequestReceived,
					Verb:       "watch",
					RequestURI: longRunningPath,
				},
				{
					Stage:          auditinternal.StageResponseStarted,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
				{
					Stage:          auditinternal.StagePanic,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 500},
				},
			},
			true,
		},
		{
			"omit RequestReceived",
			shortRunningPath,
			"GET",
			"",
			[]auditinternal.Stage{auditinternal.StageRequestReceived},
			func(w http.ResponseWriter, req *http.Request) {
				w.Write([]byte("foo"))
			},
			[]auditinternal.Event{
				{
					Stage:          auditinternal.StageResponseComplete,
					Verb:           "get",
					RequestURI:     shortRunningPath,
					ResponseStatus: &metav1.Status{Code: 200},
				},
			},
			true,
		},
		{
			"emit Panic only",
			longRunningPath,
			"GET",
			"",
			[]auditinternal.Stage{auditinternal.StageRequestReceived, auditinternal.StageResponseStarted, auditinternal.StageResponseComplete},
			func(w http.ResponseWriter, req *http.Request) {
				w.Write([]byte("foo"))
				panic("kaboom")
			},
			[]auditinternal.Event{
				{
					Stage:          auditinternal.StagePanic,
					Verb:           "watch",
					RequestURI:     longRunningPath,
					ResponseStatus: &metav1.Status{Code: 500},
				},
			},
			true,
		},
	} {
		t.Run(test.desc, func(t *testing.T) {
			sink := &fakeAuditSink{}
			policyChecker := policy.FakeChecker(auditinternal.LevelRequestResponse, test.omitStages)
			handler := WithAudit(http.HandlerFunc(test.handler), sink, policyChecker, func(r *http.Request, ri *request.RequestInfo) bool {
				// simplified long-running check
				return ri.Verb == "watch"
			})

			req, _ := http.NewRequest(test.verb, test.path, nil)
			req = withTestContext(req, &user.DefaultInfo{Name: "admin"}, nil)
			if test.auditID != "" {
				req.Header.Add("Audit-ID", test.auditID)
			}
			req.RemoteAddr = "127.0.0.1"

			func() {
				defer func() {
					recover()
				}()
				handler.ServeHTTP(httptest.NewRecorder(), req)
			}()

			events := sink.Events()
			t.Logf("audit log: %v", events)

			if len(events) != len(test.expected) {
				t.Fatalf("Unexpected amount of lines in audit log: %d", len(events))
			}
			expectedID := types.UID("")
			for i, expect := range test.expected {
				event := events[i]
				if "admin" != event.User.Username {
					t.Errorf("Unexpected username: %s", event.User.Username)
				}
				if event.Stage != expect.Stage {
					t.Errorf("Unexpected Stage: %s", event.Stage)
				}
				if event.Verb != expect.Verb {
					t.Errorf("Unexpected Verb: %s", event.Verb)
				}
				if event.RequestURI != expect.RequestURI {
					t.Errorf("Unexpected RequestURI: %s", event.RequestURI)
				}

				if test.auditID != "" && event.AuditID != types.UID(test.auditID) {
					t.Errorf("Unexpected AuditID in audit event, AuditID should be the same with Audit-ID http header")
				}
				if expectedID == types.UID("") {
					expectedID = event.AuditID
				} else if expectedID != event.AuditID {
					t.Errorf("Audits for one request should share the same AuditID, %s differs from %s", expectedID, event.AuditID)
				}
				if event.ObjectRef.APIVersion != "v1" {
					t.Errorf("Unexpected apiVersion: %s", event.ObjectRef.APIVersion)
				}
				if event.ObjectRef.APIGroup != "" {
					t.Errorf("Unexpected apiGroup: %s", event.ObjectRef.APIGroup)
				}
				if (event.ResponseStatus == nil) != (expect.ResponseStatus == nil) {
					t.Errorf("Unexpected ResponseStatus: %v", event.ResponseStatus)
					continue
				}
				if (event.ResponseStatus != nil) && (event.ResponseStatus.Code != expect.ResponseStatus.Code) {
					t.Errorf("Unexpected status code : %d", event.ResponseStatus.Code)
				}
			}
		})
	}
}

func TestAuditNoPanicOnNilUser(t *testing.T) {
	policyChecker := policy.FakeChecker(auditinternal.LevelRequestResponse, nil)
	handler := WithAudit(&fakeHTTPHandler{}, &fakeAuditSink{}, policyChecker, nil)
	req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
	req = withTestContext(req, nil, nil)
	req.RemoteAddr = "127.0.0.1"
	handler.ServeHTTP(httptest.NewRecorder(), req)
}

func TestAuditLevelNone(t *testing.T) {
	sink := &fakeAuditSink{}
	var handler http.Handler
	handler = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(200)
	})
	policyChecker := policy.FakeChecker(auditinternal.LevelNone, nil)
	handler = WithAudit(handler, sink, policyChecker, nil)

	req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
	req.RemoteAddr = "127.0.0.1"
	req = withTestContext(req, &user.DefaultInfo{Name: "admin"}, nil)

	handler.ServeHTTP(httptest.NewRecorder(), req)
	if len(sink.events) > 0 {
		t.Errorf("Generated events, but should not have: %#v", sink.events)
	}
}

func TestAuditIDHttpHeader(t *testing.T) {
	for _, test := range []struct {
		desc           string
		requestHeader  string
		level          auditinternal.Level
		expectedHeader bool
	}{
		{
			"no http header when there is no audit",
			"",
			auditinternal.LevelNone,
			false,
		},
		{
			"no http header when there is no audit even the request header specified",
			uuid.New().String(),
			auditinternal.LevelNone,
			false,
		},
		{
			"server generated header",
			"",
			auditinternal.LevelRequestResponse,
			true,
		},
		{
			"user provided header",
			uuid.New().String(),
			auditinternal.LevelRequestResponse,
			true,
		},
	} {
		sink := &fakeAuditSink{}
		var handler http.Handler
		handler = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(200)
		})
		policyChecker := policy.FakeChecker(test.level, nil)
		handler = WithAudit(handler, sink, policyChecker, nil)

		req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
		req.RemoteAddr = "127.0.0.1"
		req = withTestContext(req, &user.DefaultInfo{Name: "admin"}, nil)
		if test.requestHeader != "" {
			req.Header.Add("Audit-ID", test.requestHeader)
		}

		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		resp := w.Result()
		if test.expectedHeader {
			if resp.Header.Get("Audit-ID") == "" {
				t.Errorf("[%s] expected Audit-ID http header returned, but not returned", test.desc)
				continue
			}
			// if get Audit-ID returned, it should be the same with the requested one
			if test.requestHeader != "" && resp.Header.Get("Audit-ID") != test.requestHeader {
				t.Errorf("[%s] returned audit http header is not the same with the requested http header, expected: %s, get %s", test.desc, test.requestHeader, resp.Header.Get("Audit-ID"))
			}
		} else {
			if resp.Header.Get("Audit-ID") != "" {
				t.Errorf("[%s] expected no Audit-ID http header returned, but got %s", test.desc, resp.Header.Get("Audit-ID"))
			}
		}
	}
}

func withTestContext(req *http.Request, user user.Info, audit *auditinternal.Event) *http.Request {
	ctx := req.Context()
	if user != nil {
		ctx = request.WithUser(ctx, user)
	}
	if audit != nil {
		ctx = request.WithAuditEvent(ctx, audit)
	}
	if info, err := newTestRequestInfoResolver().NewRequestInfo(req); err == nil {
		ctx = request.WithRequestInfo(ctx, info)
	}
	return req.WithContext(ctx)
}

var backendCrt1 = []byte(`-----BEGIN CERTIFICATE-----
MIIDTjCCAjagAwIBAgIJANYWBFaLyBC/MA0GCSqGSIb3DQEBCwUAMFAxCzAJBgNV
BAYTAlBMMQ8wDQYDVQQIDAZQb2xhbmQxDzANBgNVBAcMBkdkYW5zazELMAkGA1UE
CgwCU0sxEjAQBgNVBAMMCTEyNy4wLjAuMTAeFw0yMDEyMTExMDI0MzBaFw0zMDEy
MDkxMDI0MzBaMFAxCzAJBgNVBAYTAlBMMQ8wDQYDVQQIDAZQb2xhbmQxDzANBgNV
BAcMBkdkYW5zazELMAkGA1UECgwCU0sxEjAQBgNVBAMMCTEyNy4wLjAuMTCCASIw
DQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMYax2q/m/N237UFMFKZsox4EyKq
De+mbaRGeKqnI7Gi9Ai3b7BPCIa7RFJ2ntpGUd5GyL+HCQHG8/f6DjsbUuhZnmn7
F7ZJeih2DP2acKkODdGbXA52kABCMdDs2DMYhR2UwECY2t+DLpxqJqE2ab8pI9Xd
BZ3pCNodS03yHXzfeJV44lCjxoDOi9ynXLjd3w3+FowomHMEBunTepiqnbgoYtnn
RW9tQyQQK5g6+/j/O1M8o71s/0loBT3vKSqNSrdlMOEGrj4yyL/Cw1NmQf1V1sGf
w1QAW5xk7Br5oh8h1D+oflGWV3Y3zluuZQnA9D+vFpjL0969oFedsgr4UU8CAwEA
AaMrMCkwCQYDVR0TBAIwADALBgNVHQ8EBAMCBaAwDwYDVR0RBAgwBocEfwAAATAN
BgkqhkiG9w0BAQsFAAOCAQEAWbOF7TOfGiC59S50okfcS7M4gwz2kcbqOftWzcA1
lT1qX6TWj7A4bVIOMAFK2tWNd4Omk6bnIAxTJdHB7b1hrBjkpt2krEGH1S8xeRRz
Gs62KQwehM3fMhLvYSEqOQMETZn9AjEigYm6ohCO5obG9Gkfz7uvuv9rbIetbAmm
YE9HdDv6qhCqtynpP2yad3v53idlrDnCIe9e4eKUD5uR/MIp9mEFgnMXR1m43/ya
DnmddSsjtzamVvI/+2Cqjb8qT8dMHZrCBK64UwSaJsUKzSeF6yNvZKQ1yfA/NrfV
P6gNULDOqtPgXFP4j+Z402gjYox1bGHjeDHh1OVSnr9jVw==
-----END CERTIFICATE-----`)

var backendKey1 = []byte(`-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDGGsdqv5vzdt+1
BTBSmbKMeBMiqg3vpm2kRniqpyOxovQIt2+wTwiGu0RSdp7aRlHeRsi/hwkBxvP3
+g47G1LoWZ5p+xe2SXoodgz9mnCpDg3Rm1wOdpAAQjHQ7NgzGIUdlMBAmNrfgy6c
aiahNmm/KSPV3QWd6QjaHUtN8h1833iVeOJQo8aAzovcp1y43d8N/haMKJhzBAbp
03qYqp24KGLZ50VvbUMkECuYOvv4/ztTPKO9bP9JaAU97ykqjUq3ZTDhBq4+Msi/
wsNTZkH9VdbBn8NUAFucZOwa+aIfIdQ/qH5Rlld2N85brmUJwPQ/rxaYy9PevaBX
nbIK+FFPAgMBAAECggEAKmdZABR7gSWUxN6TdVrIySB6mBTmXsG0/lDHS1/zV/aV
XbhGA+sm3BABk9UoM3iR1Y45MiXpW6QGXLH9kdFLccidC/pfHPmlWDvMlAwWyVjk
xFUI41+leyiwGRRZQrag57ALZshRMT6XH4vpMODAydY4gXKJ3T8gUe+rSsfkX/Hl
Ce59c8pDsV3NDy4WKy00lYZfTqBqHu10qy9W8/eVYf+RUt53nrygCesnFfmJx/P8
GnHnN06QbZdpgVgbU49u+BujkjFgKH/60Ct9A19o34upXvkPOaKbABZ4dL1lUrbo
e3L3vnSdgXh1oOsy/JyICmDG5M2b68h33YNa+qUEgQKBgQDs1rf1+hw75o7iDlnx
E46CPC+9DkDuisWLgbUyW5KHPgropPl80uqnRxmaWpYGU/Fgyml08orpduHIWxtU
0tMRKm2HoFRM010fAp3xWc/B4pt2pdRMMSjMle//4FmoNlcJ8+owmD+2eook9Qjm
qN1UsQllkSoH4zx4iI+HhDJnHwKBgQDWIdGmlZqaYGhsndkco9yK+gve6W80ik4J
qnjnv9ux28SBrlORn2zzfGcu5LkJw8Dp9yjZzVUiFT8VFsWVNNuJyFba227Qxrwz
Hb/qvd5l2DfXHk4poyMZThzg7cxkxlVaWUIBMoGynDxQZIOypc6WmTeEG5+9W4+w
NCuTKt6/0QKBgQCOgALftUUXpXmC+i+TpbixE5WFovXekRCbB8gGLKLVTLczk0+p
kx4s19LH1Ik/9XHeUutwuh5qqmTfMDIZr1/fjC+q0wTl1KbK6cAuX2NpvPbdRJmf
3lQ2BGELC+nmFAv6qQ/XfUOYf9JuuiBI6IGDW6HTwqwPYuIXg9MYLqpE8QKBgA/2
2YCH6szTnzVp10PxW4Ho/nWSBb5vCT5jPTxZ63EpJ09bxdM3hZHplm/CkaEOvRU0
XhFO46f02Y0i83waQrvU+dS7Q1nBV0qgTyybFzeUlSUulzk3dmhukGycjf59YuOn
f+pC77R3PW/o7oClJ+/GYIMy5AfkCaRjX1RLf+vhAoGBANJBi0ARkhwOWbnD2urA
0tPMURSYIZ+JW7ghMspbm1XV1NTreCB/llLNqUGQ7zLAmH+KyqJK8O37/oh3VHrV
6jp9pqrqmibtGEIpQi4D9IM8Zo9mc8GexCf0x+11mamC+ZXjT+bvLQzbcJGnG5CL
W+S7SneWTL09leh5ATNhog6s
-----END PRIVATE KEY-----`)

// The default `ResponseWriter.Write` (https://golang.org/pkg/net/http/#ResponseWriter) method's buffer is 4096 K.
// That means that most of the time `Write` method won't return any error because it will write to memory instead of sending data over the network to a client.
// Unfortunately, that means that the audit subsystem can contain misleading entries suggesting that data was successfully sent to a client whereas due to a network error sending might have failed.
//
// TestClientDisconnected tests exactly that scenario.
func TestClientDisconnected(t *testing.T) {
	clientFinished := make(chan struct{})
	defer close(clientFinished)
	backendFinished := make(chan struct{})
	defer close(backendFinished)
	var backendWriteError error

	// set up the backend server's handler
	auditSink := &fakeAuditSink{}
	withFakeUser := func(nextHandler http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctx := r.Context()
			ctx = request.WithUser(ctx, &user.DefaultInfo{Name: "admin"})
			if info, err := newTestRequestInfoResolver().NewRequestInfo(r); err == nil {
				ctx = request.WithRequestInfo(ctx, info)
			}

			nextHandler.ServeHTTP(w, r.WithContext(ctx))
		})
	}
	var backendHandler http.Handler
	backendHandler = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-clientFinished
		buff := make([]byte, 4096+1) // default http net library buffer is 4096
		rand.Read(buff)
		_, backendWriteError = w.Write(buff)
		backendFinished <- struct{}{}
	})
	policyChecker := policy.FakeChecker(auditinternal.LevelRequestResponse, nil)
	backendHandler = WithAudit(backendHandler, auditSink, policyChecker, nil)
	backendHandler = withFakeUser(backendHandler)
	// set up the backend server
	backendServer := httptest.NewUnstartedServer(backendHandler)
	backendCert, err := tls.X509KeyPair(backendCrt1, backendKey1)
	if err != nil {
		t.Fatalf("backend: invalid x509/key pair: %v", err)
	}
	backendServer.TLS = &tls.Config{
		Certificates: []tls.Certificate{backendCert},
		NextProtos:   []string{http2.NextProtoTLS},
	}
	backendServer.StartTLS()
	defer backendServer.Close()

	// set up the client
	clientCACertPool := x509.NewCertPool()
	clientCACertPool.AppendCertsFromPEM(backendCrt1)
	clientTLSConfig := &tls.Config{
		RootCAs:    clientCACertPool,
		NextProtos: []string{http2.NextProtoTLS},
	}
	client := &http.Client{}
	client.Transport = &http2.Transport{
		TLSClientConfig: clientTLSConfig,
	}

	// client request
	sendRequest := func() error {
		ctx, _ := context.WithTimeout(context.TODO(), 1*time.Millisecond)
		req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("https://127.0.0.1:%d", backendServer.Listener.Addr().(*net.TCPAddr).Port), nil)
		if err != nil {
			return err
		}
		_, err = client.Do(req)
		if err == nil {
			return fmt.Errorf("expected to get error - ctx cancelled")
		}
		clientFinished <- struct{}{}
		return nil
	}

	// act
	if err := sendRequest(); err != nil {
		t.Fatal(err)
	}

	// validate
	<-backendFinished
	if backendWriteError == nil {
		t.Error("expected to get an error - client disconnected before write started")

	} else if backendWriteError.Error() != "http2: stream closed" {
		t.Errorf("unexpected error received: %v", backendWriteError)
	}

	if len(auditSink.Events()) != 2 {
		t.Fatalf("unexpected events recorded in the audit log, expected exaclty 2 got %d", len(auditSink.events))
	}
	auditEntryValidated := false
	for _, ev := range auditSink.Events() {
		if ev.Stage == auditinternal.StageResponseComplete {
			if ev.ResponseStatus.Message != "error while sending data to the client, err = http2: stream closed" {
				t.Fatalf("unexpected Message received = %q", ev.ResponseStatus.Message)
			}
			if ev.ResponseStatus.Status != "Failure" {
				t.Fatalf("unexpected Status received = %q", ev.ResponseStatus.Status)
			}
			auditEntryValidated = true
		}
	}
	if !auditEntryValidated {
		t.Fatal("auditlog wasn't validated")
	}
}
