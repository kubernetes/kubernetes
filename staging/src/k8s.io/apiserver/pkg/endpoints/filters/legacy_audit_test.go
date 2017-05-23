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
	"bytes"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
)

func TestLegacyConstructResponseWriter(t *testing.T) {
	actual := legacyDecorateResponseWriter(&simpleResponseWriter{}, ioutil.Discard, "")
	switch v := actual.(type) {
	case *legacyAuditResponseWriter:
	default:
		t.Errorf("Expected auditResponseWriter, got %v", reflect.TypeOf(v))
	}

	actual = legacyDecorateResponseWriter(&fancyResponseWriter{}, ioutil.Discard, "")
	switch v := actual.(type) {
	case *fancyLegacyResponseWriterDelegator:
	default:
		t.Errorf("Expected fancyResponseWriterDelegator, got %v", reflect.TypeOf(v))
	}
}

func TestLegacyAudit(t *testing.T) {
	var buf bytes.Buffer

	handler := WithLegacyAudit(&fakeHTTPHandler{}, &fakeRequestContextMapper{
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

func TestLegacyAuditNoPanicOnNilUser(t *testing.T) {
	var buf bytes.Buffer

	handler := WithLegacyAudit(&fakeHTTPHandler{}, &fakeRequestContextMapper{}, &buf)

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
