/*
Copyright The Kubernetes Authors.

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

package utils

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestInterceptionRuleMatches(t *testing.T) {
	req, _ := http.NewRequest(http.MethodGet, "/apis/coordination.k8s.io/v1/namespaces/kube-node-lease/leases/di-target-node", nil)
	watchReq, _ := http.NewRequest(http.MethodGet, "/apis/coordination.k8s.io/v1/namespaces/kube-node-lease/leases/di-target-node?watch=true", nil)

	resolver := &request.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}

	tests := []struct {
		name     string
		rule     InterceptionRule
		request  *http.Request
		expected bool
	}{
		{
			name: "exact match",
			rule: InterceptionRule{
				Method:    http.MethodGet,
				Group:     "coordination.k8s.io",
				Resource:  "leases",
				Namespace: "kube-node-lease",
				Name:      "di-target-node",
				IsWatch:   false,
			},
			request:  req,
			expected: true,
		},
		{
			name: "wildcard matches",
			rule: InterceptionRule{
				Method:    "*",
				Group:     "*",
				Resource:  "*",
				Namespace: "*",
				Name:      "*",
				IsWatch:   false,
			},
			request:  req,
			expected: true,
		},
		{
			name: "method mismatch",
			rule: InterceptionRule{
				Method:    "POST",
				Group:     "*",
				Resource:  "*",
				Namespace: "*",
				Name:      "*",
			},
			request:  req,
			expected: false,
		},
		{
			name: "watch mismatch on standard request",
			rule: InterceptionRule{
				Method:    http.MethodGet,
				Group:     "*",
				Resource:  "*",
				Namespace: "*",
				Name:      "*",
				IsWatch:   true,
			},
			request:  req,
			expected: false,
		},
		{
			name: "watch match on watch request",
			rule: InterceptionRule{
				Method:    http.MethodGet,
				Group:     "coordination.k8s.io",
				Resource:  "leases",
				Namespace: "kube-node-lease",
				Name:      "di-target-node",
				IsWatch:   true,
			},
			request:  watchReq,
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info, _ := resolver.NewRequestInfo(tt.request)
			actual := tt.rule.Matches(tt.request, info)
			if actual != tt.expected {
				t.Errorf("expected %t, got %t", tt.expected, actual)
			}
		})
	}
}

type fakeRoundTripper func(req *http.Request) (*http.Response, error)

func (f fakeRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestInterceptingTransportStandardRequest(t *testing.T) {
	req, _ := http.NewRequest(http.MethodGet, "/apis/coordination.k8s.io/v1/namespaces/kube-node-lease/leases/di-target-node", nil)

	var hitCount int32
	rule := InterceptionRule{
		Method:    http.MethodGet,
		Group:     "coordination.k8s.io",
		Resource:  "leases",
		Namespace: "kube-node-lease",
		Name:      "di-target-node",
		IsWatch:   false,
		Hook: func(req *http.Request, eventBytes []byte) {
			atomic.AddInt32(&hitCount, 1)
			if eventBytes != nil {
				t.Errorf("expected nil eventBytes for standard request, got %s", eventBytes)
			}
		},
	}

	underlying := fakeRoundTripper(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK}, nil
	})

	transport := NewInterceptingTransport(underlying, []InterceptionRule{rule})
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected StatusOK, got %d", resp.StatusCode)
	}

	if atomic.LoadInt32(&hitCount) != 1 {
		t.Errorf("expected exactly 1 hit, got %d", atomic.LoadInt32(&hitCount))
	}
}

func TestInterceptingTransportWatchRequest(t *testing.T) {
	watchReq, _ := http.NewRequest(http.MethodGet, "/apis/coordination.k8s.io/v1/namespaces/kube-node-lease/leases/di-target-node?watch=true", nil)

	blockChan := make(chan struct{})
	defer close(blockChan)

	var watchHookHits int32
	rule := InterceptionRule{
		Method:    http.MethodGet,
		Group:     "coordination.k8s.io",
		Resource:  "leases",
		Namespace: "kube-node-lease",
		Name:      "di-target-node",
		IsWatch:   true,
		Hook: func(req *http.Request, eventBytes []byte) {
			atomic.AddInt32(&watchHookHits, 1)
			if bytes.Contains(eventBytes, []byte("di-target-node")) {
				<-blockChan
			}
		},
	}

	// Raw stream responses: first line is unmatched, second line matches (blocks), third line is unmatched
	rawStream := `{"type":"ADDED","object":{"metadata":{"name":"other-node","namespace":"kube-node-lease"}}}
{"type":"ADDED","object":{"metadata":{"name":"di-target-node","namespace":"kube-node-lease"}}}
{"type":"ADDED","object":{"metadata":{"name":"another-node","namespace":"kube-node-lease"}}}
`
	underlying := fakeRoundTripper(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(rawStream)),
		}, nil
	})

	transport := NewInterceptingTransport(underlying, []InterceptionRule{rule})
	resp, err := transport.RoundTrip(watchReq)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	reader := bufio.NewReader(resp.Body)

	// 1. First line should be read immediately
	line1, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("failed to read line 1: %v", err)
	}
	if !strings.Contains(line1, "other-node") {
		t.Errorf("expected other-node, got %s", line1)
	}

	// 2. Second line read should block. Let's check asynchronously.
	readErrChan := make(chan error, 1)
	line2Chan := make(chan string, 1)
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	go func() {
		line, err := reader.ReadString('\n')
		if err != nil {
			readErrChan <- err
			return
		}
		line2Chan <- line
	}()

	// Verify it blocks by checking timeout
	select {
	case err := <-readErrChan:
		t.Fatalf("read failed prematurely: %v", err)
	case line := <-line2Chan:
		t.Fatalf("read completed prematurely: %s", line)
	case <-ctx.Done():
		t.Log("Read correctly blocked on matching watch event!")
	}

	blockChan <- struct{}{}

	select {
	case err := <-readErrChan:
		t.Fatalf("failed to read line 2: %v", err)
	case line2 := <-line2Chan:
		if !strings.Contains(line2, "di-target-node") {
			t.Errorf("expected di-target-node, got %s", line2)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for line 2 read after releasing block")
	}

	line3, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("failed to read line 3: %v", err)
	}
	if !strings.Contains(line3, "another-node") {
		t.Errorf("expected another-node, got %s", line3)
	}

	if atomic.LoadInt32(&watchHookHits) != 1 {
		t.Errorf("expected exactly 1 hit on watch hook (only 'di-target-node' contains the target name), got %d", atomic.LoadInt32(&watchHookHits))
	}
}
