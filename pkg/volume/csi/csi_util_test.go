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

package csi

import (
	"strings"
	"testing"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
)

func TestSanitizaMsg(t *testing.T) {

	tests := []struct {
		name     string
		msg      *csipb.CreateVolumeRequest
		expected string
	}{
		{
			name: "test_with_csi_secret",
			msg: &csipb.CreateVolumeRequest{
				Secrets: map[string]string{"secret1": "secret1", "secret2": "secret2"},
			},
			expected: "secrets:<key:\"secret1\" value:\"* * * Sanitized * * *\" > secrets:<key:\"secret2\" value:\"* * * Sanitized * * *\" > ",
		},
		{
			name: "test_without_csi_secret",
			msg: &csipb.CreateVolumeRequest{
				Name:       "test-volume",
				Parameters: map[string]string{"param1": "param1", "param2": "param2"},
			},
			expected: "name:\"test-volume\" parameters:<key:\"param1\" value:\"param1\" > parameters:<key:\"param2\" value:\"param2\" > ",
		},
	}

	for _, test := range tests {
		result := SanitizeMsg(test.msg)
		if c := strings.Compare(test.expected, result); c != 0 {
			t.Fatalf("Test %s failed, expected: %s got: %s, c: %d", test.name, test.expected, result, c)
		}
	}
}
