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
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	unversionedapi "k8s.io/kubernetes/pkg/api/unversioned"
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
		expectErr                      func(err error) bool
		sendErr                        error
	}{
		{
			name:            "server supports client default",
			version:         "version1",
			config:          &unversioned.Config{},
			serverVersions:  []string{"version1", testapi.Default.Version()},
			clientVersions:  []string{"version1", testapi.Default.Version()},
			expectedVersion: "version1",
		},
		{
			name:            "server falls back to client supported",
			version:         testapi.Default.Version(),
			config:          &unversioned.Config{},
			serverVersions:  []string{"version1"},
			clientVersions:  []string{"version1", testapi.Default.Version()},
			expectedVersion: "version1",
		},
		{
			name:            "explicit version supported",
			version:         "",
			config:          &unversioned.Config{Version: testapi.Default.Version()},
			serverVersions:  []string{"version1", testapi.Default.Version()},
			clientVersions:  []string{"version1", testapi.Default.Version()},
			expectedVersion: testapi.Default.Version(),
		},
		{
			name:           "explicit version not supported",
			version:        "",
			config:         &unversioned.Config{Version: testapi.Default.Version()},
			serverVersions: []string{"version1"},
			clientVersions: []string{"version1", testapi.Default.Version()},
			expectErr:      func(err error) bool { return strings.Contains(err.Error(), `server does not support API version "v1"`) },
		},
		{
			name:           "connection refused error",
			config:         &unversioned.Config{Version: testapi.Default.Version()},
			serverVersions: []string{"version1"},
			clientVersions: []string{"version1", testapi.Default.Version()},
			sendErr:        errors.New("connection refused"),
			expectErr:      func(err error) bool { return strings.Contains(err.Error(), "connection refused") },
		},
	}
	codec := testapi.Default.Codec()

	for _, test := range tests {
		fakeClient := &fake.RESTClient{
			Codec: codec,
			Resp: &http.Response{
				StatusCode: 200,
				Body:       objBody(&unversionedapi.APIVersions{Versions: test.serverVersions}),
			},
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				if test.sendErr != nil {
					return nil, test.sendErr
				}
				return &http.Response{StatusCode: 200, Body: objBody(&unversionedapi.APIVersions{Versions: test.serverVersions})}, nil
			}),
		}
		c := unversioned.NewOrDie(test.config)
		c.Client = fakeClient.Client
		response, err := unversioned.NegotiateVersion(c, test.config, test.version, test.clientVersions)
		if err == nil && test.expectErr != nil {
			t.Errorf("expected error, got nil for [%s].", test.name)
		}
		if err != nil {
			if test.expectErr == nil || !test.expectErr(err) {
				t.Errorf("unexpected error for [%s]: %v.", test.name, err)
			}
			continue
		}
		if response != test.expectedVersion {
			t.Errorf("expected version %s, got %s.", test.expectedVersion, response)
		}
	}
}
