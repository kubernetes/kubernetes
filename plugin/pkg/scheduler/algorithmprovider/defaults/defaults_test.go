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

package defaults

import (
	"os"
	"testing"
)

func TestGetMaxVols(t *testing.T) {
	previousValue := os.Getenv(KubeMaxPDVols)
	defaultValue := 39

	tests := []struct {
		rawMaxVols string
		expected   int
		test       string
	}{
		{
			rawMaxVols: "invalid",
			expected:   defaultValue,
			test:       "Unable to parse maximum PD volumes value, using default value",
		},
		{
			rawMaxVols: "-2",
			expected:   defaultValue,
			test:       "Maximum PD volumes must be a positive value, using default value",
		},
		{
			rawMaxVols: "40",
			expected:   40,
			test:       "Parse maximum PD volumes value from env",
		},
	}

	for _, test := range tests {
		os.Setenv(KubeMaxPDVols, test.rawMaxVols)
		result := getMaxVols(defaultValue)
		if result != test.expected {
			t.Errorf("%s: expected %v got %v", test.test, test.expected, result)
		}
	}

	os.Unsetenv(KubeMaxPDVols)
	if previousValue != "" {
		os.Setenv(KubeMaxPDVols, previousValue)
	}
}
