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
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
)

func TestPropogatingPanic(t *testing.T) {
	var buf bytes.Buffer
	klog.SetOutput(&buf)
	klog.LogToStderr(false)
	defer klog.LogToStderr(true)

	panicMsg := "panic as designed"
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic(panicMsg)
	})
	resolver := &request.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}
	ts := httptest.NewServer(WithRoutine(WithPanicRecovery(handler, resolver), func(_ *http.Request, _ *request.RequestInfo) bool { return true }))
	defer ts.Close()
	_, err := http.Get(ts.URL)
	if err == nil {
		t.Error("expected to receive an error")
	}

	klog.Flush()
	klog.SetOutput(&bytes.Buffer{}) // prevent further writes into buf
	capturedOutput := buf.String()

	if !strings.Contains(capturedOutput, panicMsg) || !strings.Contains(capturedOutput, "apiserver panic'd") {
		t.Errorf("unexpected out captured actual = %v", capturedOutput)
	}
}

func TestExecutionWithRoutine(t *testing.T) {
	var executed bool
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t := TaskFrom(r.Context())
		t.Func = func() {
			executed = true
		}
	})
	ts := httptest.NewServer(WithRoutine(handler, func(_ *http.Request, _ *request.RequestInfo) bool { return true }))
	defer ts.Close()

	_, err := http.Get(ts.URL)
	if err != nil {
		t.Errorf("got unexpected error on request: %v", err)
	}
	if !executed {
		t.Error("expected to execute")
	}
}
