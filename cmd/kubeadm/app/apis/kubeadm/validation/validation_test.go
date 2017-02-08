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

package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateServiceSubnet(t *testing.T) {
	var tests = []struct {
		s        string
		f        *field.Path
		expected bool
	}{
		{"", nil, false},
		{"this is not a cidr", nil, false}, // not a CIDR
		{"10.0.0.1", nil, false},           // not a CIDR
		{"10.96.0.1/29", nil, false},       // CIDR too small, only 8 addresses and we require at least 10
		{"10.96.0.1/28", nil, true},        // a /28 subnet is ok because it can contain 16 addresses
		{"10.96.0.1/12", nil, true},        // the default subnet should obviously pass as well
	}
	for _, rt := range tests {
		actual := ValidateServiceSubnet(rt.s, rt.f)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateServiceSubnet:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}
