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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	uapi "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/restclient/fake"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/runtime"
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
		requiredVersion *uapi.GroupVersion
		expectedVersion *uapi.GroupVersion
		serverVersions  []string
		clientVersions  []uapi.GroupVersion
		expectErr       func(err error) bool
		sendErr         error
		statusCode      int
	}{
		{
			name:            "server supports client default",
			serverVersions:  []string{"version1", registered.GroupOrDie(api.GroupName).GroupVersion.String()},
			clientVersions:  []uapi.GroupVersion{{Version: "version1"}, registered.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &uapi.GroupVersion{Version: "version1"},
			statusCode:      http.StatusOK,
		},
		{
			name:            "server falls back to client supported",
			serverVersions:  []string{"version1"},
			clientVersions:  []uapi.GroupVersion{{Version: "version1"}, registered.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &uapi.GroupVersion{Version: "version1"},
			statusCode:      http.StatusOK,
		},
		{
			name:            "explicit version supported",
			requiredVersion: &uapi.GroupVersion{Version: "v1"},
			serverVersions:  []string{"/version1", registered.GroupOrDie(api.GroupName).GroupVersion.String()},
			clientVersions:  []uapi.GroupVersion{{Version: "version1"}, registered.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &uapi.GroupVersion{Version: "v1"},
			statusCode:      http.StatusOK,
		},
		{
			name:            "explicit version not supported on server",
			requiredVersion: &uapi.GroupVersion{Version: "v1"},
			serverVersions:  []string{"version1"},
			clientVersions:  []uapi.GroupVersion{{Version: "version1"}, registered.GroupOrDie(api.GroupName).GroupVersion},
			expectErr:       func(err error) bool { return strings.Contains(err.Error(), `server does not support API version "v1"`) },
			statusCode:      http.StatusOK,
		},
		{
			name:            "explicit version not supported on client",
			requiredVersion: &uapi.GroupVersion{Version: "v1"},
			serverVersions:  []string{"v1"},
			clientVersions:  []uapi.GroupVersion{{Version: "version1"}},
			expectErr:       func(err error) bool { return strings.Contains(err.Error(), `client does not support API version "v1"`) },
			statusCode:      http.StatusOK,
		},
		{
			name:           "connection refused error",
			serverVersions: []string{"version1"},
			clientVersions: []uapi.GroupVersion{{Version: "version1"}, registered.GroupOrDie(api.GroupName).GroupVersion},
			sendErr:        errors.New("connection refused"),
			expectErr:      func(err error) bool { return strings.Contains(err.Error(), "connection refused") },
			statusCode:     http.StatusOK,
		},
		{
			name:            "discovery fails due to 403 Forbidden errors and thus serverVersions is empty, use default GroupVersion",
			clientVersions:  []uapi.GroupVersion{{Version: "version1"}, registered.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &uapi.GroupVersion{Version: "version1"},
			statusCode:      http.StatusForbidden,
		},
		{
			name:            "discovery fails due to 404 Not Found errors and thus serverVersions is empty, use requested GroupVersion",
			requiredVersion: &uapi.GroupVersion{Version: "version1"},
			clientVersions:  []uapi.GroupVersion{{Version: "version1"}, registered.GroupOrDie(api.GroupName).GroupVersion},
			expectedVersion: &uapi.GroupVersion{Version: "version1"},
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
		c := discovery.NewDiscoveryClientForConfigOrDie(&restclient.Config{})
		c.RESTClient().(*restclient.RESTClient).Client = fakeClient.Client
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
