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

package audit

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/util/sets"
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

type fakeRequestContextMapper struct{}

func (*fakeRequestContextMapper) Get(req *http.Request) (api.Context, bool) {
	return api.WithUser(api.NewContext(), &user.DefaultInfo{Name: "admin"}), true

}

func (*fakeRequestContextMapper) Update(req *http.Request, context api.Context) error {
	return nil
}

func TestAudit(t *testing.T) {
	var buf bytes.Buffer
	attributeGetter := apiserver.NewRequestAttributeGetter(&fakeRequestContextMapper{},
		&apiserver.RequestInfoResolver{APIPrefixes: sets.NewString("api", "apis"), GrouplessAPIPrefixes: sets.NewString("api")})
	handler := WithAudit(&fakeHTTPHandler{}, attributeGetter, &buf)
	req, _ := http.NewRequest("GET", "/api/v1/namespaces/default/pods", nil)
	req.RemoteAddr = "127.0.0.1"
	handler.ServeHTTP(httptest.NewRecorder(), req)
	line := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(line) != 2 {
		t.Fatalf("Unexpected amount of lines in audit log: %d", len(line))
	}
	match, err := regexp.MatchString(`[\d\:\-\.\+TZ]+ AUDIT: id="[\w-]+" ip="127.0.0.1" method="GET" user="admin" as="<self>" asgroups="<lookup>" namespace="default" uri="/api/v1/namespaces/default/pods"`, line[0])
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
