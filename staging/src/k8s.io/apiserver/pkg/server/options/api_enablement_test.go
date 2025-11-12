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
	"strings"
	"testing"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	cliflag "k8s.io/component-base/cli/flag"
)

type fakeGroupRegistry struct{}

func (f fakeGroupRegistry) IsGroupRegistered(group string) bool {
	return group == "apiregistration.k8s.io"
}

func TestAPIEnablementOptionsValidate(t *testing.T) {
	testCases := []struct {
		name          string
		runtimeConfig cliflag.ConfigurationMap
		expectErr     string
	}{
		{
			name: "test when options is nil",
		},
		{
			name:          "test when invalid runtime-config with only api/all=false",
			runtimeConfig: cliflag.ConfigurationMap{"api/all": "false"},
			expectErr:     "invalid runtime-config with only api/all=false",
		},
		{
			name:          "test when ConfigurationMap key is invalid",
			runtimeConfig: cliflag.ConfigurationMap{"apiall": "false"},
			expectErr:     "runtime-config invalid key",
		},
		{
			name:          "test when unknown api groups",
			runtimeConfig: cliflag.ConfigurationMap{"api/v1": "true", "api/v1beta2": "true"},
			expectErr:     "unknown api groups api/v1,api/v1beta2",
		},
		{
			name:          "test when valid api groups",
			runtimeConfig: cliflag.ConfigurationMap{"apiregistration.k8s.io/v1beta1": "true"},
		},
		{
			name:          "test when invalid api groups",
			runtimeConfig: cliflag.ConfigurationMap{"apiregistration.k8s.io/v1beta1": "true"},
		},
	}
	testGroupRegistry := fakeGroupRegistry{}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			testOptions := &APIEnablementOptions{
				RuntimeConfig: testcase.runtimeConfig,
			}
			errs := testOptions.Validate(testGroupRegistry)
			if len(testcase.expectErr) != 0 && !strings.Contains(utilerrors.NewAggregate(errs).Error(), testcase.expectErr) {
				t.Errorf("got err: %v, expected err: %s", errs, testcase.expectErr)
			}

			if len(testcase.expectErr) == 0 && len(errs) != 0 {
				t.Errorf("got err: %s, expected err nil", errs)
			}
		})
	}
}
