/*
Copyright The Kubernetes Authors.

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
	"fmt"
	"testing"

	serviceconfig "k8s.io/cloud-provider/controllers/service/config"
)

func TestServiceControllerValidation(t *testing.T) {
	testCases := []struct {
		desc   string
		input  *ServiceControllerOptions
		expect []error
	}{
		{
			desc: "empty options",
		},
		{
			desc:  "concurrent-service-syncs positive value",
			input: &ServiceControllerOptions{ServiceControllerConfiguration: &serviceconfig.ServiceControllerConfiguration{ConcurrentServiceSyncs: 3}},
		},
		{
			desc:   "concurrent-service-syncs zero value",
			input:  &ServiceControllerOptions{ServiceControllerConfiguration: &serviceconfig.ServiceControllerConfiguration{ConcurrentServiceSyncs: 0}},
			expect: []error{fmt.Errorf("concurrent-service-syncs must be at least 1, but got 0")},
		},
		{
			desc:   "concurrent-service-syncs negative value",
			input:  &ServiceControllerOptions{ServiceControllerConfiguration: &serviceconfig.ServiceControllerConfiguration{ConcurrentServiceSyncs: -1}},
			expect: []error{fmt.Errorf("concurrent-service-syncs must be at least 1, but got -1")},
		},
	}
	for _, tc := range testCases {
		got := tc.input.Validate()
		if !errSliceEq(tc.expect, got) {
			t.Errorf("%v: expected: %v  got: %v", tc.desc, tc.expect, got)
		}
	}
}
