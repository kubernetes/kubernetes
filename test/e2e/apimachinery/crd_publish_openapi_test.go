/*
Copyright 2026 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestWaitForOpenAPISchemaRetriesRequestErrors(t *testing.T) {
	requestCount := 0
	var requestContext context.Context
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		requestCount++
		if requestCount == 1 {
			requestContext = req.Context()
		}
		deadline, ok := req.Context().Deadline()
		firstDeadline, _ := requestContext.Deadline()
		if !ok || !deadline.Equal(firstDeadline) {
			t.Error("requests must share a polling deadline")
		}
		if requestCount == 1 {
			return nil, errors.New("temporary network failure")
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{},
			Body:       io.NopCloser(strings.NewReader(`{"swagger":"2.0"}`)),
			Request:    req,
		}, nil
	})}

	err := waitForOpenAPISchemaWithClient(client, "https://example.com/openapi/v2", time.Millisecond, time.Second, func(*spec.Swagger) (bool, string) {
		return true, ""
	})
	if err != nil {
		t.Fatalf("waitForOpenAPISchemaWithClient returned an error: %v", err)
	}
	if want := waitSuccessThreshold + 1; requestCount != want {
		t.Fatalf("expected %d requests, got %d", want, requestCount)
	}
	if err := requestContext.Err(); err != context.Canceled {
		t.Errorf("request context error after success = %v, want %v", err, context.Canceled)
	}
}

func TestWaitForOpenAPISchemaCancelsBlockedRequests(t *testing.T) {
	for name, blockBody := range map[string]bool{"response headers": false, "response body": true} {
		t.Run(name, func(t *testing.T) {
			canceled := make(chan struct{})
			var once sync.Once
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if blockBody {
					_, _ = io.WriteString(w, `{"swagger":`)
					w.(http.Flusher).Flush()
				}
				<-req.Context().Done()
				once.Do(func() { close(canceled) })
			}))
			defer server.Close()
			client := server.Client()
			client.Timeout = 5 * time.Second // Bound the test if request cancellation regresses.
			start := time.Now()
			err := waitForOpenAPISchemaWithClient(client, server.URL, time.Millisecond, 250*time.Millisecond, func(*spec.Swagger) (bool, string) {
				return true, ""
			})
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("expected polling deadline error, got %v", err)
			}
			select {
			case <-canceled:
			case <-time.After(time.Second):
				t.Fatal("server did not observe request cancellation")
			}
			if elapsed := time.Since(start); elapsed >= 2*time.Second {
				t.Fatalf("polling deadline did not interrupt the blocked request: took %v", elapsed)
			}
		})
	}
}

func TestWaitForOpenAPISchemaRequestErrorTimeout(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("temporary network failure")
	})}
	err := waitForOpenAPISchemaWithClient(client, "https://example.com/openapi/v2", time.Millisecond, 25*time.Millisecond, func(*spec.Swagger) (bool, string) {
		t.Error("predicate called after a failed request")
		return true, ""
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected polling deadline error, got %v", err)
	}
	if !strings.Contains(err.Error(), "temporary network failure") {
		t.Errorf("timeout error does not contain the last request failure: %v", err)
	}
}
