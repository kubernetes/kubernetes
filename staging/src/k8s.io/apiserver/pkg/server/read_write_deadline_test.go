/*
Copyright 2024 The Kubernetes Authors.

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

package server

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/request"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestPerHandlerReadWriteDeadlineWithNonLongRunningRequest(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PerHandlerReadWriteTimeout, true)()

	fakeAudit := &fakeAudit{}
	config, _ := setUp(t)
	config.AuditPolicyRuleEvaluator = fakeAudit
	config.AuditBackend = fakeAudit
	s, err := config.Complete(nil).New("test", NewEmptyDelegate())
	if err != nil {
		t.Fatalf("Error in setting up a GenericAPIServer object: %v", err)
	}

	longRunningFn := config.LongRunningFunc
	clientDoneCh, handlerDoneCh := make(chan struct{}), make(chan error, 1)
	timeoutWant := 1 * time.Second
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDoneCh)
		ctx := r.Context()

		reqInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			t.Errorf("expected the request context to have a RequestInfo associated")
			return
		}
		if longRunningFn(r, reqInfo) {
			t.Errorf("wrong test setup, wanted a non long-running request, but got: %#v", reqInfo)
			return
		}
		receivedAt, ok := request.ReceivedTimestampFrom(ctx)
		if !ok {
			t.Errorf("expected the request context to have a received at timestamp, but got: %s", receivedAt)
			return
		}
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Errorf("expected the request context to have a deadline")
			return
		}
		if timeoutGot := deadline.Sub(receivedAt); timeoutWant != timeoutGot {
			t.Errorf("expected the request context to have a deadline of: %s, but got: %s", timeoutWant, timeoutGot)
			return
		}

		<-clientDoneCh

		// a threshold of 10s to account for round trip and CI flakes
		sinceDeadline, threshold := time.Since(deadline), 10*time.Second
		t.Logf("client has received a response %s after deadline", sinceDeadline)
		if sinceDeadline > threshold {
			t.Errorf("expected the client to receive a response earlier, took %s", sinceDeadline)
		}

		func() {
			now := time.Now()
			count := 0
			defer func() {
				duration := time.Since(now)
				t.Logf("After timeout, Write (1KB of data) was invoked %d times, total duration before error: %s", count, duration)
				// 10s should be long enough to account for CI flakes
				if duration > threshold {
					t.Errorf("Write took too long to return a timeout error: %s", duration)
				}
			}()
			for {
				count++
				if _, err := w.Write(bytes.Repeat([]byte("a"), 1024)); err != nil {
					handlerDoneCh <- err
					break
				}
			}
		}()
	})
	s.Handler.NonGoRestfulMux.Handle("/ping", handler)

	server := httptest.NewUnstartedServer(s.Handler)
	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	client := server.Client()
	func() {
		defer close(clientDoneCh)
		_, err := client.Get(server.URL + fmt.Sprintf("/ping?timeout=%s", timeoutWant))
		if !responsewritertesting.IsStreamReadOrWriteTimeout(err) {
			t.Errorf("expected a stream reset error, but got: %v", err)
		}
	}()

	select {
	case err := <-handlerDoneCh:
		if err == nil || !strings.Contains(err.Error(), "i/o timeout") {
			t.Errorf("expected Write (invoked after deadline passes) to return a timeout error, but got: %v", err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the request handler to have terminated")
	}
}

func TestPerHandlerReadWriteDeadlineWithWatchRequest(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PerHandlerReadWriteTimeout, true)()

	fakeAudit := &fakeAudit{}
	config, _ := setUp(t)
	config.AuditPolicyRuleEvaluator = fakeAudit
	config.AuditBackend = fakeAudit
	s, err := config.Complete(nil).New("test", NewEmptyDelegate())
	if err != nil {
		t.Fatalf("Error in setting up a GenericAPIServer object: %v", err)
	}

	longRunningFn := config.LongRunningFunc
	handlerDoneCh := make(chan struct{})
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDoneCh)
		ctx := r.Context()

		reqInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			t.Errorf("expected the request context to have a RequestInfo associated")
			return
		}
		if !longRunningFn(r, reqInfo) || reqInfo.Verb != "watch" {
			t.Errorf("wrong test setup, wanted a watch request, but got: %#v", reqInfo)
			return
		}
		if receivedAt, ok := request.ReceivedTimestampFrom(ctx); !ok {
			t.Errorf("expected the request context to have a received at timestamp, but got: %s", receivedAt)
			return
		}
		if _, ok := ctx.Deadline(); ok {
			t.Errorf("did not expect the request context to have a deadline set")
			return
		}
	})

	path := "/api/v1/namespaces/ns1/resources"
	s.Handler.NonGoRestfulMux.Handle(path, handler)

	server := httptest.NewUnstartedServer(s.Handler)
	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	client := server.Client()
	if _, err := client.Get(server.URL + fmt.Sprintf("%s?timeout=1m&timeoutSeconds=60&watch=1", path)); err != nil {
		t.Errorf("expected no error from client.Get, but got: %v", err)
	}

	select {
	case <-handlerDoneCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the request handler to have terminated")
	}
}
