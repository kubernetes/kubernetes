/*
Copyright 2020 The Kubernetes Authors.

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

package kubeadm

import (
	"reflect"
	"testing"
)

func TestAPIEndpointFromString(t *testing.T) {
	var tests = []struct {
		apiEndpoint      string
		expectedEndpoint APIEndpoint
		expectedErr      bool
	}{
		{apiEndpoint: "1.2.3.4:1234", expectedEndpoint: APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234}},
		{apiEndpoint: "1.2.3.4:-1", expectedErr: true},
		{apiEndpoint: "1.2.::1234", expectedErr: true},
		{apiEndpoint: "1.2.3.4:65536", expectedErr: true},
		{apiEndpoint: "1.2.3.456:1234", expectedErr: true},
		{apiEndpoint: "[::1]:1234", expectedEndpoint: APIEndpoint{AdvertiseAddress: "::1", BindPort: 1234}},
		{apiEndpoint: "[::1]:-1", expectedErr: true},
		{apiEndpoint: "[::1]:65536", expectedErr: true},
		{apiEndpoint: "[::1:1234", expectedErr: true},
		{apiEndpoint: "[::g]:1234", expectedErr: true},
	}
	for _, rt := range tests {
		t.Run(rt.apiEndpoint, func(t *testing.T) {
			apiEndpoint, err := APIEndpointFromString(rt.apiEndpoint)
			if (err != nil) != rt.expectedErr {
				t.Errorf("expected error %v, got %v, error: %v", rt.expectedErr, err != nil, err)
			}
			if !reflect.DeepEqual(apiEndpoint, rt.expectedEndpoint) {
				t.Errorf("expected API endpoint: %v; got: %v", rt.expectedEndpoint, apiEndpoint)
			}
		})
	}
}

func TestString(t *testing.T) {
	var tests = []struct {
		name        string
		apiEndpoint APIEndpoint
		expected    string
	}{
		{name: "ipv4 and port", apiEndpoint: APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234}, expected: "1.2.3.4:1234"},
		{name: "ipv6 and port", apiEndpoint: APIEndpoint{AdvertiseAddress: "::1", BindPort: 1234}, expected: "[::1]:1234"},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			apiEndpointString := rt.apiEndpoint.String()
			if apiEndpointString != rt.expected {
				t.Errorf(
					"failed String:\n\texpected: %s\n\t  actual: %s",
					rt.expected,
					apiEndpointString,
				)
			}
		})
	}
}
