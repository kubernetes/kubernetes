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

package request

import (
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestGetAPIRequestInfo(t *testing.T) {
	namespaceAll := metav1.NamespaceAll
	successCases := []struct {
		method              string
		url                 string
		expectedVerb        string
		expectedAPIPrefix   string
		expectedAPIGroup    string
		expectedAPIVersion  string
		expectedNamespace   string
		expectedResource    string
		expectedSubresource string
		expectedName        string
		expectedParts       []string
	}{

		// resource paths
		{MethodGet, "/api/v1/namespaces", "list", "api", "", "v1", "", "namespaces", "", "", []string{"namespaces"}},
		{MethodGet, "/api/v1/namespaces/other", "get", "api", "", "v1", "other", "namespaces", "", "other", []string{"namespaces", "other"}},

		{MethodGet, "/api/v1/namespaces/other/pods", "list", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},
		{MethodGet, "/api/v1/namespaces/other/pods/foo", "get", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{MethodHead, "/api/v1/namespaces/other/pods/foo", "get", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{MethodGet, "/api/v1/pods", "list", "api", "", "v1", namespaceAll, "pods", "", "", []string{"pods"}},
		{MethodHead, "/api/v1/pods", "list", "api", "", "v1", namespaceAll, "pods", "", "", []string{"pods"}},
		{MethodGet, "/api/v1/namespaces/other/pods/foo", "get", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{MethodGet, "/api/v1/namespaces/other/pods", "list", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},

		// special verbs
		{MethodGet, "/api/v1/proxy/namespaces/other/pods/foo", "proxy", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{MethodGet, "/api/v1/proxy/namespaces/other/pods/foo/subpath/not/a/subresource", "proxy", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo", "subpath", "not", "a", "subresource"}},
		{MethodGet, "/api/v1/watch/pods", "watch", "api", "", "v1", namespaceAll, "pods", "", "", []string{"pods"}},
		{MethodGet, "/api/v1/pods?watch=true", "watch", "api", "", "v1", namespaceAll, "pods", "", "", []string{"pods"}},
		{MethodGet, "/api/v1/pods?watch=false", "list", "api", "", "v1", namespaceAll, "pods", "", "", []string{"pods"}},
		{MethodGet, "/api/v1/watch/namespaces/other/pods", "watch", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},
		{MethodGet, "/api/v1/namespaces/other/pods?watch=1", "watch", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},
		{MethodGet, "/api/v1/namespaces/other/pods?watch=0", "list", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},

		// subresource identification
		{MethodGet, "/api/v1/namespaces/other/pods/foo/status", "get", "api", "", "v1", "other", "pods", "status", "foo", []string{"pods", "foo", "status"}},
		{MethodGet, "/api/v1/namespaces/other/pods/foo/proxy/subpath", "get", "api", "", "v1", "other", "pods", "proxy", "foo", []string{"pods", "foo", "proxy", "subpath"}},
		{MethodPut, "/api/v1/namespaces/other/finalize", "update", "api", "", "v1", "other", "namespaces", "finalize", "other", []string{"namespaces", "other", "finalize"}},
		{MethodPut, "/api/v1/namespaces/other/status", "update", "api", "", "v1", "other", "namespaces", "status", "other", []string{"namespaces", "other", "status"}},

		// verb identification
		{MethodPatch, "/api/v1/namespaces/other/pods/foo", "patch", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{MethodDelete, "/api/v1/namespaces/other/pods/foo", "delete", "api", "", "v1", "other", "pods", "", "foo", []string{"pods", "foo"}},
		{MethodPost, "/api/v1/namespaces/other/pods", "create", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},

		// deletecollection verb identification
		{MethodDelete, "/api/v1/nodes", "deletecollection", "api", "", "v1", "", "nodes", "", "", []string{"nodes"}},
		{MethodDelete, "/api/v1/namespaces", "deletecollection", "api", "", "v1", "", "namespaces", "", "", []string{"namespaces"}},
		{MethodDelete, "/api/v1/namespaces/other/pods", "deletecollection", "api", "", "v1", "other", "pods", "", "", []string{"pods"}},
		{MethodDelete, "/apis/extensions/v1/namespaces/other/pods", "deletecollection", "api", "extensions", "v1", "other", "pods", "", "", []string{"pods"}},

		// api group identification
		{MethodPost, "/apis/extensions/v1/namespaces/other/pods", "create", "api", "extensions", "v1", "other", "pods", "", "", []string{"pods"}},

		// api version identification
		{MethodPost, "/apis/extensions/v1beta3/namespaces/other/pods", "create", "api", "extensions", "v1beta3", "other", "pods", "", "", []string{"pods"}},
	}

	resolver := newTestRequestInfoResolver()

	for _, successCase := range successCases {
		req, _ := http.NewRequest(successCase.method, successCase.url, nil)

		apiRequestInfo, err := resolver.NewRequestInfo(req)
		if err != nil {
			t.Errorf("Unexpected error for url: %s %v", successCase.url, err)
		}
		if !apiRequestInfo.IsResourceRequest {
			t.Errorf("Expected resource request")
		}
		if successCase.expectedVerb != apiRequestInfo.Verb {
			t.Errorf("Unexpected verb for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedVerb, apiRequestInfo.Verb)
		}
		if successCase.expectedAPIVersion != apiRequestInfo.APIVersion {
			t.Errorf("Unexpected apiVersion for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedAPIVersion, apiRequestInfo.APIVersion)
		}
		if successCase.expectedNamespace != apiRequestInfo.Namespace {
			t.Errorf("Unexpected namespace for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedNamespace, apiRequestInfo.Namespace)
		}
		if successCase.expectedResource != apiRequestInfo.Resource {
			t.Errorf("Unexpected resource for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedResource, apiRequestInfo.Resource)
		}
		if successCase.expectedSubresource != apiRequestInfo.Subresource {
			t.Errorf("Unexpected resource for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedSubresource, apiRequestInfo.Subresource)
		}
		if successCase.expectedName != apiRequestInfo.Name {
			t.Errorf("Unexpected name for url: %s, expected: %s, actual: %s", successCase.url, successCase.expectedName, apiRequestInfo.Name)
		}
		if !reflect.DeepEqual(successCase.expectedParts, apiRequestInfo.Parts) {
			t.Errorf("Unexpected parts for url: %s, expected: %v, actual: %v", successCase.url, successCase.expectedParts, apiRequestInfo.Parts)
		}
	}

	errorCases := map[string]string{
		"no resource path":            "/",
		"just apiversion":             "/api/version/",
		"just prefix, group, version": "/apis/group/version/",
		"apiversion with no resource": "/api/version/",
		"bad prefix":                  "/badprefix/version/resource",
		"missing api group":           "/apis/version/resource",
	}
	for k, v := range errorCases {
		req, err := http.NewRequest(MethodGet, v, nil)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
		}
		apiRequestInfo, err := resolver.NewRequestInfo(req)
		if err != nil {
			t.Errorf("%s: Unexpected error %v", k, err)
		}
		if apiRequestInfo.IsResourceRequest {
			t.Errorf("%s: expected non-resource request", k)
		}
	}
}

func TestGetNonAPIRequestInfo(t *testing.T) {
	tests := map[string]struct {
		url      string
		expected bool
	}{
		"simple groupless":  {"/api/version/resource", true},
		"simple group":      {"/apis/group/version/resource/name/subresource", true},
		"more steps":        {"/api/version/resource/name/subresource", true},
		"group list":        {"/apis/batch/v1/job", true},
		"group get":         {"/apis/batch/v1/job/foo", true},
		"group subresource": {"/apis/batch/v1/job/foo/scale", true},

		"bad root":                     {"/not-api/version/resource", false},
		"group without enough steps":   {"/apis/extensions/v1beta1", false},
		"group without enough steps 2": {"/apis/extensions/v1beta1/", false},
		"not enough steps":             {"/api/version", false},
		"one step":                     {"/api", false},
		"zero step":                    {"/", false},
		"empty":                        {"", false},
	}

	resolver := newTestRequestInfoResolver()

	for testName, tc := range tests {
		req, _ := http.NewRequest(MethodGet, tc.url, nil)

		apiRequestInfo, err := resolver.NewRequestInfo(req)
		if err != nil {
			t.Errorf("%s: Unexpected error %v", testName, err)
		}
		if e, a := tc.expected, apiRequestInfo.IsResourceRequest; e != a {
			t.Errorf("%s: expected %v, actual %v", testName, e, a)
		}
	}
}

func newTestRequestInfoResolver() *RequestInfoFactory {
	return &RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}
}

func TestSelectorParsing(t *testing.T) {
	tests := []struct {
		name                  string
		method                string
		url                   string
		expectedName          string
		expectedErr           error
		expectedVerb          string
		expectedFieldSelector string
		expectedLabelSelector string
	}{
		{
			name:                  "no selector",
			method:                MethodGet,
			url:                   "/apis/group/version/resource",
			expectedVerb:          "list",
			expectedFieldSelector: "",
		},
		{
			name:                  "metadata.name selector",
			method:                MethodGet,
			url:                   "/apis/group/version/resource?fieldSelector=metadata.name=name1",
			expectedName:          "name1",
			expectedVerb:          "list",
			expectedFieldSelector: "metadata.name=name1",
		},
		{
			name:                  "metadata.name selector with watch",
			method:                MethodGet,
			url:                   "/apis/group/version/resource?watch=true&fieldSelector=metadata.name=name1",
			expectedName:          "name1",
			expectedVerb:          "watch",
			expectedFieldSelector: "metadata.name=name1",
		},
		{
			name:                  "random selector",
			method:                MethodGet,
			url:                   "/apis/group/version/resource?fieldSelector=foo=bar&labelSelector=baz=qux",
			expectedName:          "",
			expectedVerb:          "list",
			expectedFieldSelector: "foo=bar",
			expectedLabelSelector: "baz=qux",
		},
		{
			name:                  "invalid selector with metadata.name",
			method:                MethodGet,
			url:                   "/apis/group/version/resource?fieldSelector=metadata.name=name1,foo",
			expectedName:          "",
			expectedErr:           fmt.Errorf("invalid selector"),
			expectedVerb:          "list",
			expectedFieldSelector: "metadata.name=name1,foo",
		},
		{
			name:                  "invalid selector with metadata.name with watch",
			method:                MethodGet,
			url:                   "/apis/group/version/resource?fieldSelector=metadata.name=name1,foo&watch=true",
			expectedName:          "",
			expectedErr:           fmt.Errorf("invalid selector"),
			expectedVerb:          "watch",
			expectedFieldSelector: "metadata.name=name1,foo",
		},
		{
			name:                  "invalid selector with metadata.name with watch false",
			method:                MethodGet,
			url:                   "/apis/group/version/resource?fieldSelector=metadata.name=name1,foo&watch=false",
			expectedName:          "",
			expectedErr:           fmt.Errorf("invalid selector"),
			expectedVerb:          "list",
			expectedFieldSelector: "metadata.name=name1,foo",
		},
		{
			name:                  "selector on deletecollection is honored",
			method:                MethodDelete,
			url:                   "/apis/group/version/resource?fieldSelector=foo=bar&labelSelector=baz=qux",
			expectedName:          "",
			expectedVerb:          "deletecollection",
			expectedFieldSelector: "foo=bar",
			expectedLabelSelector: "baz=qux",
		},
		{
			name:                  "selector on repeated param matches parsed param",
			method:                MethodGet,
			url:                   "/apis/group/version/resource?fieldSelector=metadata.name=foo&fieldSelector=metadata.name=bar&labelSelector=foo=bar&labelSelector=foo=baz",
			expectedName:          "foo",
			expectedVerb:          "list",
			expectedFieldSelector: "metadata.name=foo",
			expectedLabelSelector: "foo=bar",
		},
		{
			name:                  "selector on other verb is ignored",
			method:                MethodGet,
			url:                   "/apis/group/version/resource/name?fieldSelector=foo=bar&labelSelector=foo=bar",
			expectedName:          "name",
			expectedVerb:          "get",
			expectedFieldSelector: "",
			expectedLabelSelector: "",
		},
		{
			name:                  "selector on deprecated root type watch is not parsed",
			method:                MethodGet,
			url:                   "/apis/group/version/watch/resource?fieldSelector=metadata.name=foo&labelSelector=foo=bar",
			expectedName:          "",
			expectedVerb:          "watch",
			expectedFieldSelector: "",
			expectedLabelSelector: "",
		},
		{
			name:                  "selector on deprecated root item watch is not parsed",
			method:                MethodGet,
			url:                   "/apis/group/version/watch/resource/name?fieldSelector=metadata.name=foo&labelSelector=foo=bar",
			expectedName:          "name",
			expectedVerb:          "watch",
			expectedFieldSelector: "",
			expectedLabelSelector: "",
		},
	}

	resolver := newTestRequestInfoResolver()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AuthorizeWithSelectors, true)

	for _, tc := range tests {
		req, _ := http.NewRequest(tc.method, tc.url, nil)

		apiRequestInfo, err := resolver.NewRequestInfo(req)
		if err != nil {
			if tc.expectedErr == nil || !strings.Contains(err.Error(), tc.expectedErr.Error()) {
				t.Errorf("%s: Unexpected error %v", tc.name, err)
			}
		}
		if e, a := tc.expectedName, apiRequestInfo.Name; e != a {
			t.Errorf("%s: expected %v, actual %v", tc.name, e, a)
		}
		if e, a := tc.expectedVerb, apiRequestInfo.Verb; e != a {
			t.Errorf("%s: expected verb %v, actual %v", tc.name, e, a)
		}
		if e, a := tc.expectedFieldSelector, apiRequestInfo.FieldSelector; e != a {
			t.Errorf("%s: expected fieldSelector %v, actual %v", tc.name, e, a)
		}
		if e, a := tc.expectedLabelSelector, apiRequestInfo.LabelSelector; e != a {
			t.Errorf("%s: expected labelSelector %v, actual %v", tc.name, e, a)
		}
	}
}
