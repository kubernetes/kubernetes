/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned_test

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	unversioned_api "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
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
		name, version, expectedVersion string
		serverVersions                 []string
		clientVersions                 []string
		config                         *unversioned.Config
		expectErr                      bool
	}{
		{
			name:            "server supports client default",
			version:         "version1",
			config:          &unversioned.Config{},
			serverVersions:  []string{"version1", testapi.Default.Version()},
			clientVersions:  []string{"version1", testapi.Default.Version()},
			expectedVersion: "version1",
			expectErr:       false,
		},
		{
			name:            "server falls back to client supported",
			version:         testapi.Default.Version(),
			config:          &unversioned.Config{},
			serverVersions:  []string{"version1"},
			clientVersions:  []string{"version1", testapi.Default.Version()},
			expectedVersion: "version1",
			expectErr:       false,
		},
		{
			name:            "explicit version supported",
			version:         "",
			config:          &unversioned.Config{Version: testapi.Default.Version()},
			serverVersions:  []string{"version1", testapi.Default.Version()},
			clientVersions:  []string{"version1", testapi.Default.Version()},
			expectedVersion: testapi.Default.Version(),
			expectErr:       false,
		},
		{
			name:            "explicit version not supported",
			version:         "",
			config:          &unversioned.Config{Version: testapi.Default.Version()},
			serverVersions:  []string{"version1"},
			clientVersions:  []string{"version1", testapi.Default.Version()},
			expectedVersion: "",
			expectErr:       true,
		},
	}
	codec := testapi.Default.Codec()

	for _, test := range tests {
		fakeClient := &fake.RESTClient{
			Codec: codec,
			Resp: &http.Response{
				StatusCode: 200,
				Body:       objBody(&unversioned_api.APIVersions{Versions: test.serverVersions}),
			},
			Client: fake.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Body: objBody(&unversioned_api.APIVersions{Versions: test.serverVersions})}, nil
			}),
		}
		c := unversioned.NewOrDie(test.config)
		c.Client = fakeClient.Client
		response, err := unversioned.NegotiateVersion(c, test.config, test.version, test.clientVersions)
		if err == nil && test.expectErr {
			t.Errorf("expected error, got nil for [%s].", test.name)
		}
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error for [%s]: %v.", test.name, err)
		}
		if response != test.expectedVersion {
			t.Errorf("expected version %s, got %s.", test.expectedVersion, response)
		}
	}
}
