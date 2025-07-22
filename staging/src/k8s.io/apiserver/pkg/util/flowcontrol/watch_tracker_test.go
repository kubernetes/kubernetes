/*
Copyright 2021 The Kubernetes Authors.

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

package flowcontrol

import (
	"context"
	"net/http"
	"net/url"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func httpRequest(method, path, rawQuery string) *http.Request {
	return &http.Request{
		Method: method,
		URL: &url.URL{
			Path:     path,
			RawQuery: rawQuery,
		},
	}
}

func newWatchIdentifier(apiGroup, resource, namespace, name string) *watchIdentifier {
	return &watchIdentifier{
		apiGroup:  apiGroup,
		resource:  resource,
		namespace: namespace,
		name:      name,
	}
}

func TestRegisterWatch(t *testing.T) {
	testCases := []struct {
		name     string
		request  *http.Request
		expected *watchIdentifier
	}{
		{
			name:     "watch all objects",
			request:  httpRequest("GET", "/api/v1/pods", "watch=true"),
			expected: newWatchIdentifier("", "pods", "", ""),
		},
		{
			name:     "list all objects",
			request:  httpRequest("GET", "/api/v1/pods", ""),
			expected: nil,
		},
		{
			name:     "watch namespace-scoped objects",
			request:  httpRequest("GET", "/api/v1/namespaces/foo/pods", "watch=true"),
			expected: newWatchIdentifier("", "pods", "foo", ""),
		},
		{
			name:     "watch single object",
			request:  httpRequest("GET", "/api/v1/namespaces/foo/pods", "watch=true&fieldSelector=metadata.name=mypod"),
			expected: newWatchIdentifier("", "pods", "foo", "mypod"),
		},
		{
			name:     "watch single cluster-scoped object",
			request:  httpRequest("GET", "/api/v1/namespaces", "watch=true&fieldSelector=metadata.name=myns"),
			expected: newWatchIdentifier("", "namespaces", "", "myns"),
		},
		{
			name:     "watch all objects from api-group",
			request:  httpRequest("GET", "/apis/group/v1/pods", "watch=true"),
			expected: newWatchIdentifier("group", "pods", "", ""),
		},
		{
			name:     "watch namespace-scoped objects",
			request:  httpRequest("GET", "/apis/group/v1/namespaces/foo/pods", "watch=true"),
			expected: newWatchIdentifier("group", "pods", "foo", ""),
		},
		{
			name:     "watch single object",
			request:  httpRequest("GET", "/apis/group/v1/namespaces/foo/pods", "watch=true&fieldSelector=metadata.name=mypod"),
			expected: newWatchIdentifier("group", "pods", "foo", "mypod"),
		},
		{
			name:     "watch indexed object",
			request:  httpRequest("GET", "/apis/group/v1/namespaces/foo/pods", "watch=true&fieldSelector=spec.nodeName="),
			expected: newWatchIdentifier("group", "pods", "foo", ""),
		},
	}

	requestInfoFactory := &request.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			watchTracker := &watchTracker{
				indexes:    getBuiltinIndexes(),
				watchCount: make(map[watchIdentifier]int),
			}

			requestInfo, err := requestInfoFactory.NewRequestInfo(testCase.request)
			if err != nil {
				t.Fatalf("unexpected error from requestInfo creation: %#v", err)
			}
			ctx := request.WithRequestInfo(context.Background(), requestInfo)
			r := testCase.request.WithContext(ctx)

			forget := watchTracker.RegisterWatch(r)
			if testCase.expected == nil {
				if forget != nil {
					t.Errorf("unexpected watch registered: %#v", watchTracker.watchCount)
				}
				return
			}

			if forget == nil {
				t.Errorf("watch should be registered, got: %v", forget)
				return
			}
			if count := watchTracker.watchCount[*testCase.expected]; count != 1 {
				t.Errorf("unexpected watch registered: %#v", watchTracker.watchCount)
			}
			forget()
			if count := watchTracker.watchCount[*testCase.expected]; count != 0 {
				t.Errorf("forget should unregister the watch: %#v", watchTracker.watchCount)
			}
		})
	}
}

func TestGetInterestedWatchCount(t *testing.T) {
	watchTracker := NewWatchTracker()

	registeredWatches := []*http.Request{
		httpRequest("GET", "api/v1/pods", "watch=true"),
		httpRequest("GET", "api/v1/namespaces/foo/pods", "watch=true"),
		httpRequest("GET", "api/v1/namespaces/foo/pods", "watch=true&fieldSelector=metadata.name=mypod"),
		httpRequest("GET", "api/v1/namespaces/bar/pods", "watch=true&fieldSelector=metadata.name=mypod"),
		httpRequest("GET", "apis/group/v1/namespaces/foo/pods", "watch=true"),
		httpRequest("GET", "apis/group/v1/namespaces/bar/pods", "watch=true&fieldSelector=metadata.name=mypod"),
	}
	requestInfoFactory := &request.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}
	for _, req := range registeredWatches {
		requestInfo, err := requestInfoFactory.NewRequestInfo(req)
		if err != nil {
			t.Fatalf("unexpected error from requestInfo creation: %#v", err)
		}
		r := req.WithContext(request.WithRequestInfo(context.Background(), requestInfo))
		if forget := watchTracker.RegisterWatch(r); forget == nil {
			t.Errorf("watch wasn't registered: %#v", requestInfo)
		}
	}

	testCases := []struct {
		name     string
		request  *http.Request
		expected int
	}{
		{
			name:     "pod creation in foo namespace",
			request:  httpRequest("POST", "/api/v1/namespaces/foo/pods", ""),
			expected: 2,
		},
		{
			name:     "mypod update in foo namespace",
			request:  httpRequest("PUT", "/api/v1/namespaces/foo/pods/mypod", ""),
			expected: 3,
		},
		{
			name:     "mypod patch in foo namespace",
			request:  httpRequest("PATCH", "/api/v1/namespaces/foo/pods/mypod", ""),
			expected: 3,
		},
		{
			name:     "mypod deletion in foo namespace",
			request:  httpRequest("DELETE", "/api/v1/namespaces/foo/pods/mypod", ""),
			expected: 3,
		},
		{
			name:     "otherpod update in foo namespace",
			request:  httpRequest("PUT", "/api/v1/namespaces/foo/pods/otherpod", ""),
			expected: 2,
		},
		{
			name:     "mypod get in foo namespace",
			request:  httpRequest("GET", "/api/v1/namespaces/foo/pods/mypod", ""),
			expected: 0,
		},
		{
			name:     "pods list in foo namespace",
			request:  httpRequest("GET", "/api/v1/namespaces/foo/pods", ""),
			expected: 0,
		},
		{
			name:     "pods watch in foo namespace",
			request:  httpRequest("GET", "/api/v1/namespaces/foo/pods", "watch=true"),
			expected: 0,
		},
		{
			name:     "pods proxy in foo namespace",
			request:  httpRequest("GET", "/api/v1/proxy/namespaces/foo/pods/mypod", ""),
			expected: 0,
		},
		{
			name:     "pod creation in bar namespace",
			request:  httpRequest("POST", "/api/v1/namespaces/bar/pods", ""),
			expected: 1,
		},
		{
			name:     "mypod update in bar namespace",
			request:  httpRequest("PUT", "/api/v1/namespaces/bar/pods/mypod", ""),
			expected: 2,
		},
		{
			name:     "mypod update in foo namespace in group group",
			request:  httpRequest("PUT", "/apis/group/v1/namespaces/foo/pods/mypod", ""),
			expected: 1,
		},
		{
			name:     "otherpod update in foo namespace in group group",
			request:  httpRequest("PUT", "/apis/group/v1/namespaces/foo/pods/otherpod", ""),
			expected: 1,
		},
		{
			name:     "mypod update in var namespace in group group",
			request:  httpRequest("PUT", "/apis/group/v1/namespaces/bar/pods/mypod", ""),
			expected: 1,
		},
		{
			name:     "otherpod update in bar namespace in group group",
			request:  httpRequest("PUT", "/apis/group/v1/namespaces/bar/pods/otherpod", ""),
			expected: 0,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			requestInfo, err := requestInfoFactory.NewRequestInfo(testCase.request)
			if err != nil {
				t.Fatalf("unexpected error from requestInfo creation: %#v", err)
			}

			count := watchTracker.GetInterestedWatchCount(requestInfo)
			if count != testCase.expected {
				t.Errorf("unexpected interested watch count: %d, expected %d", count, testCase.expected)
			}
		})
	}

}

func TestGetInterestedWatchCountWithIndex(t *testing.T) {
	watchTracker := NewWatchTracker()

	registeredWatches := []*http.Request{
		httpRequest("GET", "api/v1/pods", "watch=true"),
		httpRequest("GET", "api/v1/namespaces/foo/pods", "watch=true"),
		httpRequest("GET", "api/v1/namespaces/foo/pods", "watch=true&fieldSelector=metadata.name=mypod"),
		httpRequest("GET", "api/v1/namespaces/foo/pods", "watch=true&fieldSelector=spec.nodeName="),
		// The watches below will be ignored due to index.
		httpRequest("GET", "api/v1/namespaces/foo/pods", "watch=true&fieldSelector=spec.nodeName=node1"),
		httpRequest("GET", "api/v1/namespaces/foo/pods", "watch=true&fieldSelector=spec.nodeName=node2"),
	}
	requestInfoFactory := &request.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}
	for _, req := range registeredWatches {
		requestInfo, err := requestInfoFactory.NewRequestInfo(req)
		if err != nil {
			t.Fatalf("unexpected error from requestInfo creation: %#v", err)
		}
		r := req.WithContext(request.WithRequestInfo(context.Background(), requestInfo))
		if forget := watchTracker.RegisterWatch(r); forget == nil {
			t.Errorf("watch wasn't registered: %#v", requestInfo)
		}
	}

	testCases := []struct {
		name     string
		request  *http.Request
		expected int
	}{
		{
			name:     "pod creation in foo namespace",
			request:  httpRequest("POST", "/api/v1/namespaces/foo/pods", ""),
			expected: 3,
		},
		{
			name:     "mypod update in foo namespace",
			request:  httpRequest("PUT", "/api/v1/namespaces/foo/pods/mypod", ""),
			expected: 4,
		},
		{
			name:     "mypod patch in foo namespace",
			request:  httpRequest("PATCH", "/api/v1/namespaces/foo/pods/mypod", ""),
			expected: 4,
		},
		{
			name:     "mypod deletion in foo namespace",
			request:  httpRequest("DELETE", "/api/v1/namespaces/foo/pods/mypod", ""),
			expected: 4,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			requestInfo, err := requestInfoFactory.NewRequestInfo(testCase.request)
			if err != nil {
				t.Fatalf("unexpected error from requestInfo creation: %#v", err)
			}

			count := watchTracker.GetInterestedWatchCount(requestInfo)
			if count != testCase.expected {
				t.Errorf("unexpected interested watch count: %d, expected %d", count, testCase.expected)
			}
		})
	}

}
