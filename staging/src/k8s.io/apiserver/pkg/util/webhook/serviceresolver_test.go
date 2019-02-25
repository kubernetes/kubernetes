/*
Copyright 2017 The Kubernetes Authors.

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

package webhook

import (
	"fmt"
	"testing"
)

func TestDefaultServiceResolver(t *testing.T) {
	scenarios := []struct {
		serviceName      string
		serviceNamespace string
		expectedOutput   string
		expectError      bool
	}{
		// scenario 1: a service name along with a namespace resolves
		{serviceName: "ross", serviceNamespace: "andromeda", expectedOutput: "https://ross.andromeda.svc:443"},
		// scenario 2: a service name without a namespace does not resolve
		{serviceName: "ross", expectError: true},
		// scenario 3: cannot resolve an empty service name
		{serviceNamespace: "andromeda", expectError: true},
	}

	// act
	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("scenario %d", index), func(t *testing.T) {
			target := defaultServiceResolver{}
			serviceURL, err := target.ResolveEndpoint(scenario.serviceNamespace, scenario.serviceName)

			if err != nil && !scenario.expectError {
				t.Errorf("unexpected error has occurred = %v", err)
			}
			if err == nil && scenario.expectError {
				t.Error("expected an error but got nothing")
			}
			if !scenario.expectError {
				if serviceURL.String() != scenario.expectedOutput {
					t.Errorf("expected = %s, got = %s", scenario.expectedOutput, serviceURL.String())
				}
			}
		})
	}
}
