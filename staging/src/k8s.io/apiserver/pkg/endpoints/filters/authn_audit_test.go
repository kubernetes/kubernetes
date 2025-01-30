/*
Copyright 2017 The Kubernetes Authors.

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
	"crypto/tls"
	"crypto/x509"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit/policy"
)

func TestFailedAuthnAudit(t *testing.T) {
	sink := &fakeAuditSink{}
	fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)
	handler := WithFailedAuthenticationAudit(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "", http.StatusUnauthorized)
		}),
		sink, fakeRuleEvaluator)
	req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
	req.RemoteAddr = "127.0.0.1"
	req = withTestContext(req, nil, nil)
	req.SetBasicAuth("username", "password")
	handler.ServeHTTP(httptest.NewRecorder(), req)

	if len(sink.events) != 1 {
		t.Fatalf("Unexpected number of audit events generated, expected 1, got: %d", len(sink.events))
	}
	ev := sink.events[0]
	if ev.ResponseStatus.Code != http.StatusUnauthorized {
		t.Errorf("Unexpected response code, expected unauthorized, got %d", ev.ResponseStatus.Code)
	}
	if !strings.Contains(ev.ResponseStatus.Message, "basic") {
		t.Errorf("Expected response status message to contain basic auth method, got %s", ev.ResponseStatus.Message)
	}
	if ev.Verb != "list" {
		t.Errorf("Unexpected verb, expected list, got %s", ev.Verb)
	}
	if ev.RequestURI != "/api/v1/namespaces/default/pods" {
		t.Errorf("Unexpected user, expected /api/v1/namespaces/default/pods, got %s", ev.RequestURI)
	}
}

func TestFailedMultipleAuthnAudit(t *testing.T) {
	sink := &fakeAuditSink{}
	fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)
	handler := WithFailedAuthenticationAudit(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "", http.StatusUnauthorized)
		}),
		sink, fakeRuleEvaluator)
	req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
	req.RemoteAddr = "127.0.0.1"
	req = withTestContext(req, nil, nil)
	req.SetBasicAuth("username", "password")
	req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{{}}}
	handler.ServeHTTP(httptest.NewRecorder(), req)

	if len(sink.events) != 1 {
		t.Fatalf("Unexpected number of audit events generated, expected 1, got: %d", len(sink.events))
	}
	ev := sink.events[0]
	if ev.ResponseStatus.Code != http.StatusUnauthorized {
		t.Errorf("Unexpected response code, expected unauthorized, got %d", ev.ResponseStatus.Code)
	}
	if !strings.Contains(ev.ResponseStatus.Message, "basic") || !strings.Contains(ev.ResponseStatus.Message, "x509") {
		t.Errorf("Expected response status message to contain basic and x509 auth method, got %s", ev.ResponseStatus.Message)
	}
	if ev.Verb != "list" {
		t.Errorf("Unexpected verb, expected list, got %s", ev.Verb)
	}
	if ev.RequestURI != "/api/v1/namespaces/default/pods" {
		t.Errorf("Unexpected user, expected /api/v1/namespaces/default/pods, got %s", ev.RequestURI)
	}
}

func TestFailedAuthnAuditWithoutAuthorization(t *testing.T) {
	sink := &fakeAuditSink{}
	fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, nil)
	handler := WithFailedAuthenticationAudit(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "", http.StatusUnauthorized)
		}),
		sink, fakeRuleEvaluator)
	req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
	req.RemoteAddr = "127.0.0.1"
	req = withTestContext(req, nil, nil)
	handler.ServeHTTP(httptest.NewRecorder(), req)

	if len(sink.events) != 1 {
		t.Fatalf("Unexpected number of audit events generated, expected 1, got: %d", len(sink.events))
	}
	ev := sink.events[0]
	if ev.ResponseStatus.Code != http.StatusUnauthorized {
		t.Errorf("Unexpected response code, expected unauthorized, got %d", ev.ResponseStatus.Code)
	}
	if !strings.Contains(ev.ResponseStatus.Message, "no credentials provided") {
		t.Errorf("Expected response status message to contain no credentials provided, got %s", ev.ResponseStatus.Message)
	}
	if ev.Verb != "list" {
		t.Errorf("Unexpected verb, expected list, got %s", ev.Verb)
	}
	if ev.RequestURI != "/api/v1/namespaces/default/pods" {
		t.Errorf("Unexpected user, expected /api/v1/namespaces/default/pods, got %s", ev.RequestURI)
	}
}

func TestFailedAuthnAuditOmitted(t *testing.T) {
	sink := &fakeAuditSink{}
	fakeRuleEvaluator := policy.NewFakePolicyRuleEvaluator(auditinternal.LevelRequestResponse, []auditinternal.Stage{auditinternal.StageResponseStarted})
	handler := WithFailedAuthenticationAudit(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "", http.StatusUnauthorized)
		}),
		sink, fakeRuleEvaluator)
	req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
	req.RemoteAddr = "127.0.0.1"
	req = withTestContext(req, nil, nil)
	handler.ServeHTTP(httptest.NewRecorder(), req)

	if len(sink.events) != 0 {
		t.Fatalf("Unexpected number of audit events generated, expected 0, got: %d", len(sink.events))
	}
}
