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

package pods

import (
	"testing"
)

func TestConvertDownwardAPIFieldLabel(t *testing.T) {
	testCases := []struct {
		version       string
		label         string
		value         string
		expectedErr   bool
		expectedLabel string
		expectedValue string
	}{
		{
			version:     "v2",
			label:       "metadata.name",
			value:       "test-pod",
			expectedErr: true,
		},
		{
			version:     "v1",
			label:       "invalid-label",
			value:       "value",
			expectedErr: true,
		},
		{
			version:       "v1",
			label:         "metadata.name",
			value:         "test-pod",
			expectedLabel: "metadata.name",
			expectedValue: "test-pod",
		},
		{
			version:       "v1",
			label:         "metadata.annotations",
			value:         "myValue",
			expectedLabel: "metadata.annotations",
			expectedValue: "myValue",
		},
		{
			version:       "v1",
			label:         "metadata.annotations['myKey']",
			value:         "myValue",
			expectedLabel: "metadata.annotations['myKey']",
			expectedValue: "myValue",
		},
		{
			version:       "v1",
			label:         "spec.host",
			value:         "127.0.0.1",
			expectedLabel: "spec.nodeName",
			expectedValue: "127.0.0.1",
		},
	}
	for _, tc := range testCases {
		label, value, err := ConvertDownwardAPIFieldLabel(tc.version, tc.label, tc.value)
		if err != nil {
			if tc.expectedErr {
				continue
			}
			t.Errorf("ConvertDownwardAPIFieldLabel(%s, %s, %s) failed: %s",
				tc.version, tc.label, tc.value, err)
		}
		if tc.expectedLabel != label || tc.expectedValue != value {
			t.Errorf("ConvertDownwardAPIFieldLabel(%s, %s, %s) = (%s, %s, nil), expected (%s, %s, nil)",
				tc.version, tc.label, tc.value, label, value, tc.expectedLabel, tc.expectedValue)
		}
	}
}
