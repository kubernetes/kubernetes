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

package modes

import "testing"

func TestIsValidAuthorizationMode(t *testing.T) {
	var tests = []struct {
		authzMode string
		expected  bool
	}{
		{"", false},
		{"rBAC", false},        // not supported
		{"falsy value", false}, // not supported
		{"RBAC", true},         // supported
		{"ABAC", true},         // supported
		{"Webhook", true},      // supported
		{"AlwaysAllow", true},  // supported
		{"AlwaysDeny", true},   // supported
	}
	for _, rt := range tests {
		actual := IsValidAuthorizationMode(rt.authzMode)
		if actual != rt.expected {
			t.Errorf(
				"failed ValidAuthorizationMode:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				actual,
			)
		}
	}
}
