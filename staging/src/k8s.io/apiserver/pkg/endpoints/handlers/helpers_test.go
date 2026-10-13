/*
Copyright 2019 The Kubernetes Authors.

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

package handlers

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestLazyTruncatedUserAgent(t *testing.T) {
	req := &http.Request{}
	req.Header = http.Header{}

	ua := "short-agent"
	req.Header.Set("User-Agent", ua)
	uaNotTruncated := &lazyTruncatedUserAgent{req}
	assert.Equal(t, ua, fmt.Sprintf("%v", uaNotTruncated))

	ua = ""
	for i := 0; i < maxUserAgentLength*2; i++ {
		ua = ua + "a"
	}
	req.Header.Set("User-Agent", ua)
	uaTruncated := &lazyTruncatedUserAgent{req}
	assert.NotEqual(t, ua, fmt.Sprintf("%v", uaTruncated))

	usUnknown := &lazyTruncatedUserAgent{}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", usUnknown))
}

func TestLazyClientIP(t *testing.T) {
	req := &http.Request{}
	req.Header = http.Header{}

	ip := "127.0.0.1"
	req.Header.Set("X-Forwarded-For", ip)

	clientIPWithReq := &lazyClientIP{req}
	assert.Equal(t, ip, fmt.Sprintf("%v", clientIPWithReq))

	clientIPWithoutReq := &lazyClientIP{}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", clientIPWithoutReq))
}

func TestLazyAccept(t *testing.T) {
	req := &http.Request{}
	req.Header = http.Header{}

	accept := "application/json"
	req.Header.Set("Accept", accept)

	acceptWithReq := &lazyAccept{req}
	assert.Equal(t, accept, fmt.Sprintf("%v", acceptWithReq))

	acceptWithoutReq := &lazyAccept{}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", acceptWithoutReq))
}

func TestLazyVerb(t *testing.T) {
	assert.Equal(t, "unknown", fmt.Sprintf("%v", &lazyVerb{}))

	u, _ := url.Parse("?watch=true")
	req := &http.Request{Method: request.MethodGet, URL: u}
	verbWithReq := &lazyVerb{req: req}
	assert.Equal(t, request.MethodWatch, fmt.Sprintf("%v", verbWithReq))
}

func TestLazyApiGroup(t *testing.T) {
	assert.Equal(t, "unknown", fmt.Sprintf("%v", &lazyAPIGroup{}))

	scopeWithEmptyReq := &lazyAPIGroup{&http.Request{}}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", scopeWithEmptyReq))

	req := &http.Request{}
	ctx := request.WithRequestInfo(context.TODO(), &request.RequestInfo{APIGroup: "apps"})
	scopeWithReq := &lazyAPIGroup{req: req.WithContext(ctx)}
	assert.Equal(t, "apps", fmt.Sprintf("%v", scopeWithReq))
}

func TestLazyApiVersion(t *testing.T) {
	assert.Equal(t, "unknown", fmt.Sprintf("%v", &lazyAPIVersion{}))

	scopeWithEmptyReq := &lazyAPIVersion{&http.Request{}}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", scopeWithEmptyReq))

	req := &http.Request{}
	ctx := request.WithRequestInfo(context.TODO(), &request.RequestInfo{APIVersion: "v1"})
	scopeWithReq := &lazyAPIVersion{req: req.WithContext(ctx)}
	assert.Equal(t, "v1", fmt.Sprintf("%v", scopeWithReq))
}

func TestLazyName(t *testing.T) {
	assert.Equal(t, "unknown", fmt.Sprintf("%v", &lazyName{}))

	scopeWithEmptyReq := &lazyName{&http.Request{}}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", scopeWithEmptyReq))

	req := &http.Request{}
	ctx := request.WithRequestInfo(context.TODO(), &request.RequestInfo{Name: "jaeger-76d45d6876-vqp8t"})
	scopeWithReq := &lazyName{req: req.WithContext(ctx)}
	assert.Equal(t, "jaeger-76d45d6876-vqp8t", fmt.Sprintf("%v", scopeWithReq))
}

func TestLazySubresource(t *testing.T) {
	assert.Equal(t, "unknown", fmt.Sprintf("%v", &lazySubresource{}))

	scopeWithEmptyReq := &lazySubresource{&http.Request{}}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", scopeWithEmptyReq))

	req := &http.Request{}
	ctx := request.WithRequestInfo(context.TODO(), &request.RequestInfo{Subresource: "binding"})
	scopeWithReq := &lazySubresource{req: req.WithContext(ctx)}
	assert.Equal(t, "binding", fmt.Sprintf("%v", scopeWithReq))
}

func TestLazyNamespace(t *testing.T) {
	assert.Equal(t, "unknown", fmt.Sprintf("%v", &lazyNamespace{}))

	scopeWithEmptyReq := &lazyNamespace{&http.Request{}}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", scopeWithEmptyReq))

	req := &http.Request{}
	ctx := request.WithRequestInfo(context.TODO(), &request.RequestInfo{Namespace: "jaeger"})
	scopeWithReq := &lazyNamespace{req: req.WithContext(ctx)}
	assert.Equal(t, "jaeger", fmt.Sprintf("%v", scopeWithReq))
}

func TestLazyResource(t *testing.T) {
	assert.Equal(t, "unknown", fmt.Sprintf("%v", &lazyResource{}))

	resourceWithEmptyReq := &lazyResource{&http.Request{}}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", resourceWithEmptyReq))

	req := &http.Request{}
	ctx := request.WithRequestInfo(context.TODO(), &request.RequestInfo{Resource: "resource"})
	resourceWithReq := &lazyResource{req: req.WithContext(ctx)}
	assert.Equal(t, "resource", fmt.Sprintf("%v", resourceWithReq))
}

func TestLazyScope(t *testing.T) {
	assert.Equal(t, "unknown", fmt.Sprintf("%v", &lazyScope{}))

	scopeWithEmptyReq := &lazyScope{&http.Request{}}
	assert.Equal(t, "unknown", fmt.Sprintf("%v", scopeWithEmptyReq))

	req := &http.Request{}
	ctx := request.WithRequestInfo(context.TODO(), &request.RequestInfo{Namespace: "ns"})
	scopeWithReq := &lazyScope{req: req.WithContext(ctx)}
	assert.Equal(t, "namespace", fmt.Sprintf("%v", scopeWithReq))
}
