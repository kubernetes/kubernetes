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

package deviceattribute

import (
	"reflect"
	"strings"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
)

func TestGetMdevUUIDAttribute(t *testing.T) {
	validUUID := "123e4567-e89b-12d3-a456-426614174000"
	expectedAttribute := DeviceAttribute{
		Name:  StandardDeviceAttributeMdevUUID,
		Value: resourceapi.DeviceAttribute{StringValue: &validUUID},
	}

	tests := map[string]struct {
		mdevUUID          string
		expectedAttribute *DeviceAttribute
		expectsError      bool
		expectedErrMsg    string
	}{
		"valid lowercase UUID": {
			mdevUUID:          validUUID,
			expectedAttribute: &expectedAttribute,
			expectsError:      false,
		},
		"valid uppercase UUID (normalized to lowercase)": {
			mdevUUID:          strings.ToUpper(validUUID),
			expectedAttribute: &expectedAttribute,
			expectsError:      false,
		},
		"invalid empty mdev UUID": {
			mdevUUID:          "",
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    "mdev UUID cannot be empty",
		},
		"invalid mdev UUID format (no hyphens)": {
			mdevUUID:          "123e4567e89b12d3a456426614174000",
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    "invalid mdev UUID format: 123e4567e89b12d3a456426614174000",
		},
		"invalid mdev UUID format (incorrect character)": {
			mdevUUID:          "123e4567-e89b-12d3-a456-42661417400g",
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    "invalid mdev UUID format: 123e4567-e89b-12d3-a456-42661417400g",
		},
		"invalid mdev UUID format (incorrect segment lengths)": {
			mdevUUID:          "123e4567-e89b-12d3-a4564-26614174000",
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    "invalid mdev UUID format: 123e4567-e89b-12d3-a4564-26614174000",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := GetMdevUUIDAttribute(test.mdevUUID)
			if test.expectsError {
				if err == nil {
					t.Errorf("Expected error but got none")
					return
				}
				if !strings.Contains(err.Error(), test.expectedErrMsg) {
					t.Errorf("Expected error message to contain %q, got %q", test.expectedErrMsg, err.Error())
					return
				}
				return
			}
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, *test.expectedAttribute) {
				t.Errorf("Expected attribute %v, got %v", test.expectedAttribute, got)
			}
		})
	}
}
