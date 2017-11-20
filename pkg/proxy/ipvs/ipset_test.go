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

package ipvs

import (
	"testing"
)

func TestCheckIPSetVersion(t *testing.T) {
	testCases := []struct {
		vstring string
		valid   bool
	}{
		// version less than "6.0" is not valid.
		{"4.0", false},
		{"5.1", false},
		{"5.1.2", false},
		// "7" is not a valid version string.
		{"7", false},
		{"6.0", true},
		{"6.1", true},
		{"6.19", true},
		{"7.0", true},
		{"8.1.2", true},
		{"9.3.4.0", true},
		{"total junk", false},
	}

	for i := range testCases {
		valid := checkMinVersion(testCases[i].vstring)
		if testCases[i].valid != valid {
			t.Errorf("Expected result: %v, Got result: %v", testCases[i].valid, valid)
		}
	}
}
