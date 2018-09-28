/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
)

func TestToAuthenticationRequestHeaderConfig(t *testing.T) {
	testCases := []struct {
		name         string
		testOptions  *RequestHeaderAuthenticationOptions
		expectConfig *authenticatorfactory.RequestHeaderConfig
	}{
		{
			name: "test when ClientCAFile is nil",
			testOptions: &RequestHeaderAuthenticationOptions{
				UsernameHeaders:     []string{"x-remote-user"},
				GroupHeaders:        []string{"x-remote-group"},
				ExtraHeaderPrefixes: []string{"x-remote-extra-"},
				AllowedNames:        []string{"kube-aggregator"},
			},
		},
		{
			name: "test when ClientCAFile is not nil",
			testOptions: &RequestHeaderAuthenticationOptions{
				ClientCAFile:        "/testClientCAFile",
				UsernameHeaders:     []string{"x-remote-user"},
				GroupHeaders:        []string{"x-remote-group"},
				ExtraHeaderPrefixes: []string{"x-remote-extra-"},
				AllowedNames:        []string{"kube-aggregator"},
			},
			expectConfig: &authenticatorfactory.RequestHeaderConfig{
				UsernameHeaders:     []string{"x-remote-user"},
				GroupHeaders:        []string{"x-remote-group"},
				ExtraHeaderPrefixes: []string{"x-remote-extra-"},
				ClientCA:            "/testClientCAFile",
				AllowedClientNames:  []string{"kube-aggregator"},
			},
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			resultConfig := testcase.testOptions.ToAuthenticationRequestHeaderConfig()
			if !reflect.DeepEqual(resultConfig, testcase.expectConfig) {
				t.Errorf("got RequestHeaderConfig: %#v, expected RequestHeaderConfig: %#v", resultConfig, testcase.expectConfig)
			}
		})
	}
}
