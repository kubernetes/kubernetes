/*
Copyright 2025 The Kubernetes Authors.

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

package http

import (
	"net/http"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func BenchmarkNewRequestForHTTPGetAction(b *testing.B) {
	httpGet := &v1.HTTPGetAction{
		Port: intstr.FromInt32(8080),
		Path: "/healthz",
		Host: "10.0.0.1",
		HTTPHeaders: []v1.HTTPHeader{
			{Name: "X-Custom-Header", Value: "test-value"},
		},
	}
	container := &v1.Container{
		Name: "test-container",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := NewRequestForHTTPGetAction(httpGet, container, "10.0.0.1", "probe")
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
}

func BenchmarkNewRequestForHTTPGetActionReuse(b *testing.B) {
	httpGet := &v1.HTTPGetAction{
		Port: intstr.FromInt32(8080),
		Path: "/healthz",
		Host: "10.0.0.1",
		HTTPHeaders: []v1.HTTPHeader{
			{Name: "X-Custom-Header", Value: "test-value"},
		},
	}
	container := &v1.Container{
		Name: "test-container",
	}

	req, err := NewRequestForHTTPGetAction(httpGet, container, "10.0.0.1", "probe")
	if err != nil {
		b.Fatalf("unexpected error creating request: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = req
	}
}

func BenchmarkNewRequestForHTTPGetActionWithHeaders(b *testing.B) {
	httpGet := &v1.HTTPGetAction{
		Port: intstr.FromInt32(8080),
		Path: "/healthz",
		Host: "10.0.0.1",
		HTTPHeaders: []v1.HTTPHeader{
			{Name: "X-Custom-Header-1", Value: "value-1"},
			{Name: "X-Custom-Header-2", Value: "value-2"},
			{Name: "X-Custom-Header-3", Value: "value-3"},
			{Name: "Authorization", Value: "Bearer token123"},
		},
	}
	container := &v1.Container{
		Name: "test-container",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := NewRequestForHTTPGetAction(httpGet, container, "10.0.0.1", "probe")
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
}

func BenchmarkNewRequestForHTTPGetActionWithHeadersReuse(b *testing.B) {
	httpGet := &v1.HTTPGetAction{
		Port: intstr.FromInt32(8080),
		Path: "/healthz",
		Host: "10.0.0.1",
		HTTPHeaders: []v1.HTTPHeader{
			{Name: "X-Custom-Header-1", Value: "value-1"},
			{Name: "X-Custom-Header-2", Value: "value-2"},
			{Name: "X-Custom-Header-3", Value: "value-3"},
			{Name: "Authorization", Value: "Bearer token123"},
		},
	}
	container := &v1.Container{
		Name: "test-container",
	}

	req, err := NewRequestForHTTPGetAction(httpGet, container, "10.0.0.1", "probe")
	if err != nil {
		b.Fatalf("unexpected error creating request: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = req
	}
}

func BenchmarkNewRequestForHTTPGetActionHTTPS(b *testing.B) {
	httpsScheme := v1.URISchemeHTTPS
	httpGet := &v1.HTTPGetAction{
		Port:   intstr.FromInt32(443),
		Path:   "/ready",
		Host:   "10.0.0.1",
		Scheme: &httpsScheme,
	}
	container := &v1.Container{
		Name: "test-container",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := NewRequestForHTTPGetAction(httpGet, container, "10.0.0.1", "probe")
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
}

func BenchmarkNewRequestForHTTPGetActionLazyCaching(b *testing.B) {
	httpGet := &v1.HTTPGetAction{
		Port: intstr.FromInt32(8080),
		Path: "/healthz",
		Host: "10.0.0.1",
		HTTPHeaders: []v1.HTTPHeader{
			{Name: "X-Custom-Header", Value: "test-value"},
		},
	}
	container := &v1.Container{
		Name: "test-container",
	}

	var cachedReq *http.Request
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if cachedReq == nil {
			req, err := NewRequestForHTTPGetAction(httpGet, container, "10.0.0.1", "probe")
			if err != nil {
				b.Fatalf("unexpected error: %v", err)
			}
			cachedReq = req
		}
		_ = cachedReq
	}
}