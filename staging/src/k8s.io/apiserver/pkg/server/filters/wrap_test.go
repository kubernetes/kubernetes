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

package filters

import (
	"bytes"
	klogtest "k8s.io/klog/v2/test"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/routine"
	"k8s.io/klog/v2"
)

func TestPropogatingPanicLongRunning(t *testing.T) {
	panicMsg := "panic as designed"
	capturedOutput := runHandlerWithPanic(t, true, panicMsg)

	if !strings.Contains(capturedOutput, panicMsg) || !strings.Contains(capturedOutput, "apiserver panic'd") {
		t.Errorf("unexpected out captured actual = %v", capturedOutput)
	}
}

func TestPropogatingPanicNotLongRunning(t *testing.T) {
	panicMsg := "panic as designed"
	capturedOutput := runHandlerWithPanic(t, false, panicMsg)

	if !strings.Contains(capturedOutput, panicMsg) || !strings.Contains(capturedOutput, "apiserver panic'd") {
		t.Errorf("unexpected out captured actual = %v", capturedOutput)
	}
}

func TestSuppressErrAbortHandlerPanic(t *testing.T) {
	capturedOutput := runHandlerWithPanic(t, true, http.ErrAbortHandler)

	if !strings.Contains(capturedOutput, "Ignoring ErrAbortHandler") ||
		strings.Contains(capturedOutput, "apiserver panic'd") {
		t.Errorf("unexpected out captured actual = %v", capturedOutput)
	}
}

func runHandlerWithPanic(t *testing.T, longRunning bool, panicVal any) string {
	buf := captureKlog(t)

	// Panic with the special sigil value http.ErrAbortHandler.  This
	// should result in no log for a long-running request.
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic(panicVal)
	})
	resolver := &request.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}

	longRunningFn := func(r *http.Request, requestInfo *request.RequestInfo) bool {
		return longRunning
	}
	ts := httptest.NewServer(routine.WithRoutine(WithPanicRecovery(handler, resolver, longRunningFn), longRunningFn))
	defer ts.Close()
	_, err := http.Get(ts.URL)
	if err == nil {
		t.Error("expected to receive an error")
	}

	klog.Flush()
	klog.SetOutput(&bytes.Buffer{}) // prevent further writes into buf
	capturedOutput := buf.String()
	return capturedOutput
}

func captureKlog(t *testing.T) *bytes.Buffer {
	klogtest.InitKlog(t)
	var buf bytes.Buffer
	klog.SetOutput(&buf)
	return &buf
}
