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
	"context"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/routine"
	"k8s.io/klog/v2"
	klogtest "k8s.io/klog/v2/test"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestPropagatingPanicLongRunning(t *testing.T) {
	panicMsg := "panic as designed"
	capturedOutput := runHandlerCaptureOutput(t, longRunning, context.Background(), func(http.ResponseWriter, *http.Request) {
		panic(panicMsg)
	})

	if !strings.Contains(capturedOutput, panicMsg) || !strings.Contains(capturedOutput, "apiserver panic'd") {
		t.Errorf("unexpected out captured actual = %v", capturedOutput)
	}
}

func TestPropagatingPanicNotLongRunning(t *testing.T) {
	panicMsg := "panic as designed"
	capturedOutput := runHandlerCaptureOutput(t, notLongRunning, context.Background(), func(http.ResponseWriter, *http.Request) {
		panic(panicMsg)
	})

	if !strings.Contains(capturedOutput, panicMsg) || !strings.Contains(capturedOutput, "apiserver panic'd") {
		t.Errorf("unexpected out captured actual = %v", capturedOutput)
	}
}

func TestSuppressClientSideAbort(t *testing.T) {
	clientCtx, cancelClientConnection := context.WithCancel(context.Background())
	defer cancelClientConnection()

	capturedOutput := runHandlerCaptureOutput(t, longRunning, clientCtx, func(_ http.ResponseWriter, req *http.Request) {
		// Simulate the client closing the connection.
		cancelClientConnection()

		// Wait for that cancelation to propagate to the server
		select {
		case <-req.Context().Done():
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for cancelation to propagate")
		}

		// Simulate (for example) ReverseProxy detecting the client closed connection.
		panic(http.ErrAbortHandler)
	})

	if !strings.Contains(capturedOutput, "suppressing timeout log") ||
		strings.Contains(capturedOutput, "apiserver panic'd") {
		t.Errorf("unexpected out captured actual = %v", capturedOutput)
	}
}

func TestPropagateServerSideAbort(t *testing.T) {
	capturedOutput := runHandlerCaptureOutput(t, longRunning, context.Background(), func(_ http.ResponseWriter, req *http.Request) {
		// Simulate (for example) ReverseProxy detecting an upstream failure.
		panic(http.ErrAbortHandler)
	})

	if strings.Contains(capturedOutput, "suppressing timeout log") ||
		strings.Contains(capturedOutput, "panic'd") ||
		!strings.Contains(capturedOutput, "Timeout or abort") {
		t.Errorf("unexpected out captured actual = %v", capturedOutput)
	}
}

func runHandlerCaptureOutput(t *testing.T, longRunning request.LongRunningRequestCheck, clientCtx context.Context, handler http.HandlerFunc) string {
	logBuf := captureKlog(t)

	resolver := &request.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}

	ts := httptest.NewServer(
		routine.WithRoutine(
			WithPanicRecovery(handler, resolver, longRunning),
			longRunning,
		),
	)
	defer ts.Close()

	req, err := http.NewRequest(http.MethodGet, ts.URL, nil)
	if err != nil {
		t.Fatal("Failed to create request", err)
	}
	req = req.Clone(clientCtx)
	_, err = http.DefaultClient.Do(req)
	if err == nil {
		t.Error("expected to receive an error")
	}

	// Wait for the server-side processing to finish so that all logs must have
	// been emitted.  Close() is idempotent.
	ts.Close()

	klog.Flush()
	klog.SetOutput(&bytes.Buffer{}) // prevent further writes into logBuf
	capturedOutput := logBuf.String()

	return capturedOutput
}

func longRunning(r *http.Request, requestInfo *request.RequestInfo) bool {
	return true
}

func notLongRunning(r *http.Request, requestInfo *request.RequestInfo) bool {
	return false
}

func captureKlog(t *testing.T) *bytes.Buffer {
	klogtest.InitKlog(t)
	var buf bytes.Buffer
	klog.SetOutput(&buf)
	return &buf
}
