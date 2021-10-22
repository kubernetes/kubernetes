package tracing

// Copyright 2018 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"context"
	"net/http"
	"testing"
)

func TestDisabled(t *testing.T) {
	if IsEnabled() {
		t.Fatal("unexpected enabled tracing")
	}
	if tr := NewTransport(&http.Transport{}); tr != nil {
		t.Fatal("unexpected non-nil transport")
	}
	if ctx := StartSpan(context.Background(), "foo"); ctx != context.Background() {
		t.Fatal("contexts don't match")
	}
}

func TestEnabled(t *testing.T) {
	mt := mockTracer{}
	Register(&mt)
	if !IsEnabled() {
		t.Fatal("unexpected disabled tracing")
	}
	if tr := NewTransport(&http.Transport{}); tr != http.DefaultTransport {
		t.Fatal("didn't receive expected transport")
	}
	ctx := StartSpan(context.Background(), "foo")
	v := ctx.Value(mockTracer{})
	if val, ok := v.(string); !ok {
		t.Fatal("unexpected value type")
	} else if val != "foo" {
		t.Fatal("unexpected value")
	}
	EndSpan(ctx, http.StatusOK, nil)
	if !mt.ended {
		t.Fatal("EndSpan didn't forward call to registered tracer")
	}
}

type mockTracer struct {
	ended bool
}

func (m mockTracer) NewTransport(base *http.Transport) http.RoundTripper {
	return http.DefaultTransport
}

func (m mockTracer) StartSpan(ctx context.Context, name string) context.Context {
	return context.WithValue(ctx, mockTracer{}, name)
}

func (m *mockTracer) EndSpan(ctx context.Context, httpStatusCode int, err error) {
	m.ended = true
}
