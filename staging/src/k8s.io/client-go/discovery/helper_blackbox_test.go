/*
Copyright 2015 The Kubernetes Authors.

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

package discovery_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	uapi "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/testapi"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
)

func objBody(object interface{}) io.ReadCloser {
	output, err := json.MarshalIndent(object, "", "")
	if err != nil {
		panic(err)
	}
	return ioutil.NopCloser(bytes.NewReader([]byte(output)))
}

func TestNegotiateVersion(t *testing.T) {
	tests := []struct {
		name            string
		requiredVersion *schema.GroupVersion
		expectedVersion *schema.GroupVersion
		serverVersions  []string
		clientVersions  []schema.GroupVersion
		expectErr       func(err error) bool
		sendErr         error
		statusCode      int
	}{
		{
			name:            "server supports client default",
			serverVersions:  []string{"version1", api.Registry.GroupOrDie(api.GroupName).GroupVersion.String()},
			clientVersions:  []schema.GroupVersion{{Version: "version1"}, api.Registry.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &schema.GroupVersion{Version: "version1"},
			statusCode:      http.StatusOK,
		},
		{
			name:            "server falls back to client supported",
			serverVersions:  []string{"version1"},
			clientVersions:  []schema.GroupVersion{{Version: "version1"}, api.Registry.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &schema.GroupVersion{Version: "version1"},
			statusCode:      http.StatusOK,
		},
		{
			name:            "explicit version supported",
			requiredVersion: &schema.GroupVersion{Version: "v1"},
			serverVersions:  []string{"/version1", api.Registry.GroupOrDie(api.GroupName).GroupVersion.String()},
			clientVersions:  []schema.GroupVersion{{Version: "version1"}, api.Registry.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &schema.GroupVersion{Version: "v1"},
			statusCode:      http.StatusOK,
		},
		{
			name:            "explicit version not supported on server",
			requiredVersion: &schema.GroupVersion{Version: "v1"},
			serverVersions:  []string{"version1"},
			clientVersions:  []schema.GroupVersion{{Version: "version1"}, api.Registry.GroupOrDie(api.GroupName).GroupVersion},
			expectErr:       func(err error) bool { return strings.Contains(err.Error(), `server does not support API version "v1"`) },
			statusCode:      http.StatusOK,
		},
		{
			name:            "explicit version not supported on client",
			requiredVersion: &schema.GroupVersion{Version: "v1"},
			serverVersions:  []string{"v1"},
			clientVersions:  []schema.GroupVersion{{Version: "version1"}},
			expectErr:       func(err error) bool { return strings.Contains(err.Error(), `client does not support API version "v1"`) },
			statusCode:      http.StatusOK,
		},
		{
			name:           "connection refused error",
			serverVersions: []string{"version1"},
			clientVersions: []schema.GroupVersion{{Version: "version1"}, api.Registry.GroupOrDie(api.GroupName).GroupVersion},
			sendErr:        errors.New("connection refused"),
			expectErr:      func(err error) bool { return strings.Contains(err.Error(), "connection refused") },
			statusCode:     http.StatusOK,
		},
		{
			name:            "discovery fails due to 403 Forbidden errors and thus serverVersions is empty, use default GroupVersion",
			clientVersions:  []schema.GroupVersion{{Version: "version1"}, api.Registry.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &schema.GroupVersion{Version: "version1"},
			statusCode:      http.StatusForbidden,
		},
		{
			name:            "discovery fails due to 404 Not Found errors and thus serverVersions is empty, use requested GroupVersion",
			requiredVersion: &schema.GroupVersion{Version: "version1"},
			clientVersions:  []schema.GroupVersion{{Version: "version1"}, api.Registry.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &schema.GroupVersion{Version: "version1"},
			statusCode:      http.StatusNotFound,
		},
		{
			name:       "discovery fails due to 403 Forbidden errors and thus serverVersions is empty, no fallback GroupVersion",
			expectErr:  func(err error) bool { return strings.Contains(err.Error(), "failed to negotiate an api version;") },
			statusCode: http.StatusForbidden,
		},
	}

	for _, test := range tests {
		fakeClient := &fake.RESTClient{
			NegotiatedSerializer: testapi.Default.NegotiatedSerializer(),
			Resp: &http.Response{
				StatusCode: test.statusCode,
				Body:       objBody(&uapi.APIVersions{Versions: test.serverVersions}),
			},
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				if test.sendErr != nil {
					return nil, test.sendErr
				}
				header := http.Header{}
				header.Set("Content-Type", runtime.ContentTypeJSON)
				return &http.Response{StatusCode: test.statusCode, Header: header, Body: objBody(&uapi.APIVersions{Versions: test.serverVersions})}, nil
			}),
		}
		c := discovery.NewDiscoveryClientForConfigOrDie(&rest.Config{})
		c.RESTClient().(*rest.RESTClient).Client = fakeClient.Client
		response, err := discovery.NegotiateVersion(c, test.requiredVersion, test.clientVersions)
		if err == nil && test.expectErr != nil {
			t.Errorf("expected error, got nil for [%s].", test.name)
		}
		if err != nil {
			if test.expectErr == nil || !test.expectErr(err) {
				t.Errorf("unexpected error for [%s]: %v.", test.name, err)
			}
			continue
		}
		if *response != *test.expectedVersion {
			t.Errorf("%s: expected version %s, got %s.", test.name, test.expectedVersion, response)
		}
	}
}

func TestFilteredBy(t *testing.T) {
	all := discovery.ResourcePredicateFunc(func(gv string, r *metav1.APIResource) bool {
		return true
	})
	none := discovery.ResourcePredicateFunc(func(gv string, r *metav1.APIResource) bool {
		return false
	})
	onlyV2 := discovery.ResourcePredicateFunc(func(gv string, r *metav1.APIResource) bool {
		return strings.HasSuffix(gv, "/v2") || gv == "v2"
	})
	onlyBar := discovery.ResourcePredicateFunc(func(gv string, r *metav1.APIResource) bool {
		return r.Kind == "Bar"
	})

	foo := []*metav1.APIResourceList{
		{
			GroupVersion: "foo/v1",
			APIResources: []metav1.APIResource{
				{Name: "bar", Kind: "Bar"},
				{Name: "test", Kind: "Test"},
			},
		},
		{
			GroupVersion: "foo/v2",
			APIResources: []metav1.APIResource{
				{Name: "bar", Kind: "Bar"},
				{Name: "test", Kind: "Test"},
			},
		},
		{
			GroupVersion: "foo/v3",
			APIResources: []metav1.APIResource{},
		},
	}

	tests := []struct {
		input             []*metav1.APIResourceList
		pred              discovery.ResourcePredicate
		expectedResources []string
	}{
		{nil, all, []string{}},
		{[]*metav1.APIResourceList{
			{GroupVersion: "foo/v1"},
		}, all, []string{}},
		{foo, all, []string{"foo/v1.bar", "foo/v1.test", "foo/v2.bar", "foo/v2.test"}},
		{foo, onlyV2, []string{"foo/v2.bar", "foo/v2.test"}},
		{foo, onlyBar, []string{"foo/v1.bar", "foo/v2.bar"}},
		{foo, none, []string{}},
	}
	for i, test := range tests {
		filtered := discovery.FilteredBy(test.pred, test.input)

		if expected, got := sets.NewString(test.expectedResources...), sets.NewString(stringify(filtered)...); !expected.Equal(got) {
			t.Errorf("[%d] unexpected group versions: expected=%v, got=%v", i, test.expectedResources, stringify(filtered))
		}
	}
}

func stringify(rls []*metav1.APIResourceList) []string {
	result := []string{}
	for _, rl := range rls {
		for _, r := range rl.APIResources {
			result = append(result, rl.GroupVersion+"."+r.Name)
		}
		if len(rl.APIResources) == 0 {
			result = append(result, rl.GroupVersion)
		}
	}
	return result
}
