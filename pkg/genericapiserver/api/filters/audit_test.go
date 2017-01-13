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
	"bytes"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/request"
)

type simpleResponseWriter struct {
	http.ResponseWriter
}

func (*simpleResponseWriter) WriteHeader(code int) {}

type fancyResponseWriter struct {
	simpleResponseWriter
}

func (*fancyResponseWriter) CloseNotify() <-chan bool { return nil }

func (*fancyResponseWriter) Flush() {}

func (*fancyResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) { return nil, nil, nil }

func TestConstructResponseWriter(t *testing.T) {
	actual := decorateResponseWriter(&simpleResponseWriter{}, ioutil.Discard, "")
	switch v := actual.(type) {
	case *auditResponseWriter:
	default:
		t.Errorf("Expected auditResponseWriter, got %v", reflect.TypeOf(v))
	}

	actual = decorateResponseWriter(&fancyResponseWriter{}, ioutil.Discard, "")
	switch v := actual.(type) {
	case *fancyResponseWriterDelegator:
	default:
		t.Errorf("Expected fancyResponseWriterDelegator, got %v", reflect.TypeOf(v))
	}
}

type fakeHTTPHandler struct{}

func (*fakeHTTPHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(200)
}

func TestAudit(t *testing.T) {
	var buf bytes.Buffer

	handler := WithAudit(&fakeHTTPHandler{}, &fakeRequestContextMapper{
		user: &user.DefaultInfo{Name: "admin"},
	}, &buf)

	req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
	req.RemoteAddr = "127.0.0.1"
	handler.ServeHTTP(httptest.NewRecorder(), req)
	line := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(line) != 2 {
		t.Fatalf("Unexpected amount of lines in audit log: %d", len(line))
	}
	match, err := regexp.MatchString(`[\d\:\-\.\+TZ]+ AUDIT: id="[\w-]+" ip="127.0.0.1" method="GET" user="admin" groups="<none>" as="<self>" asgroups="<lookup>" namespace="default" uri="/api/v1/namespaces/default/pods"`, line[0])
	if err != nil {
		t.Errorf("Unexpected error matching first line: %v", err)
	}
	if !match {
		t.Errorf("Unexpected first line of audit: %s", line[0])
	}
	match, err = regexp.MatchString(`[\d\:\-\.\+TZ]+ AUDIT: id="[\w-]+" response="200"`, line[1])
	if err != nil {
		t.Errorf("Unexpected error matching second line: %v", err)
	}
	if !match {
		t.Errorf("Unexpected second line of audit: %s", line[1])
	}
}

type fakeRequestContextMapper struct {
	user *user.DefaultInfo
}

func (m *fakeRequestContextMapper) Get(req *http.Request) (request.Context, bool) {
	ctx := request.NewContext()
	if m.user != nil {
		ctx = request.WithUser(ctx, m.user)
	}

	resolver := newTestRequestInfoResolver()
	info, err := resolver.NewRequestInfo(req)
	if err == nil {
		ctx = request.WithRequestInfo(ctx, info)
	}

	return ctx, true
}

func (*fakeRequestContextMapper) Update(req *http.Request, context request.Context) error {
	return nil
}

func TestAuditNoPanicOnNilUser(t *testing.T) {
	var buf bytes.Buffer

	handler := WithAudit(&fakeHTTPHandler{}, &fakeRequestContextMapper{}, &buf)

	req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
	req.RemoteAddr = "127.0.0.1"
	handler.ServeHTTP(httptest.NewRecorder(), req)
	line := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(line) != 2 {
		t.Fatalf("Unexpected amount of lines in audit log: %d", len(line))
	}
	match, err := regexp.MatchString(`[\d\:\-\.\+TZ]+ AUDIT: id="[\w-]+" ip="127.0.0.1" method="GET" user="<none>" groups="<none>" as="<self>" asgroups="<lookup>" namespace="default" uri="/api/v1/namespaces/default/pods"`, line[0])
	if err != nil {
		t.Errorf("Unexpected error matching first line: %v", err)
	}
	if !match {
		t.Errorf("Unexpected first line of audit: %s", line[0])
	}
	match, err = regexp.MatchString(`[\d\:\-\.\+TZ]+ AUDIT: id="[\w-]+" response="200"`, line[1])
	if err != nil {
		t.Errorf("Unexpected error matching second line: %v", err)
	}
	if !match {
		t.Errorf("Unexpected second line of audit: %s", line[1])
	}
}
