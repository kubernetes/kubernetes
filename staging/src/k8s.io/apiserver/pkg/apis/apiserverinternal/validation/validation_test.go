/*
Copyright 2020 The Kubernetes Authors.

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
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/apiserverinternal"
)

func TestValidateServerStorageVersion(t *testing.T) {
	cases := []struct {
		ssv         apiserverinternal.ServerStorageVersion
		expectedErr string
	}{
		{
			ssv: apiserverinternal.ServerStorageVersion{
				APIServerID:       "-fea",
				EncodingVersion:   "v1alpha1",
				DecodableVersions: []string{"v1alpha1", "v1"},
			},
			expectedErr: "apiServerID: Invalid value",
		},
		{
			ssv: apiserverinternal.ServerStorageVersion{
				APIServerID:       "fea",
				EncodingVersion:   "v1alpha1",
				DecodableVersions: []string{"v1beta1", "v1"},
			},
			expectedErr: "decodableVersions must include encodingVersion",
		},
		{
			ssv: apiserverinternal.ServerStorageVersion{
				APIServerID:       "fea",
				EncodingVersion:   "v1alpha1",
				DecodableVersions: []string{"v1alpha1", "v1", "-fea"},
			},
			expectedErr: "decodableVersions[2]: Invalid value",
		},
		{
			ssv: apiserverinternal.ServerStorageVersion{
				APIServerID:       "fea",
				EncodingVersion:   "v1alpha1",
				DecodableVersions: []string{"v1alpha1", "v1"},
			},
			expectedErr: "",
		},
	}

	for _, tc := range cases {
		err := validateServerStorageVersion(tc.ssv, field.NewPath("")).ToAggregate()
		if err == nil && len(tc.expectedErr) == 0 {
			continue
		}
		if err != nil && len(tc.expectedErr) == 0 {
			t.Errorf("unexpected error %v", err)
			continue
		}
		if err == nil && len(tc.expectedErr) != 0 {
			t.Errorf("unexpected empty error")
			continue
		}
		if !strings.Contains(err.Error(), tc.expectedErr) {
			t.Errorf("expected error to contain %s, got %s", tc.expectedErr, err)
		}
	}
}

func TestValidateCommonVersion(t *testing.T) {
	cases := []struct {
		status      apiserverinternal.StorageVersionStatus
		expectedErr string
	}{
		{
			status: apiserverinternal.StorageVersionStatus{
				StorageVersions:       []apiserverinternal.ServerStorageVersion{},
				CommonEncodingVersion: func() *string { a := "v1alpha1"; return &a }(),
			},
			expectedErr: "should be nil if servers do not agree on the same encoding version, or if there is no server reporting the supported versions yet",
		},
		{
			status: apiserverinternal.StorageVersionStatus{
				StorageVersions: []apiserverinternal.ServerStorageVersion{
					{
						APIServerID:     "1",
						EncodingVersion: "v1alpha1",
					},
					{
						APIServerID:     "2",
						EncodingVersion: "v1",
					},
				},
				CommonEncodingVersion: func() *string { a := "v1alpha1"; return &a }(),
			},
			expectedErr: "should be nil if servers do not agree on the same encoding version, or if there is no server reporting the supported versions yet",
		},
		{
			status: apiserverinternal.StorageVersionStatus{
				StorageVersions: []apiserverinternal.ServerStorageVersion{
					{
						APIServerID:     "1",
						EncodingVersion: "v1alpha1",
					},
					{
						APIServerID:     "2",
						EncodingVersion: "v1alpha1",
					},
				},
				CommonEncodingVersion: nil,
			},
			expectedErr: "Invalid value: \"null\": the common encoding version is v1alpha1",
		},
		{
			status: apiserverinternal.StorageVersionStatus{
				StorageVersions: []apiserverinternal.ServerStorageVersion{
					{
						APIServerID:     "1",
						EncodingVersion: "v1alpha1",
					},
					{
						APIServerID:     "2",
						EncodingVersion: "v1alpha1",
					},
				},
				CommonEncodingVersion: func() *string { a := "v1"; return &a }(),
			},
			expectedErr: "Invalid value: \"v1\": the actual common encoding version is v1alpha1",
		},
		{
			status: apiserverinternal.StorageVersionStatus{
				StorageVersions: []apiserverinternal.ServerStorageVersion{
					{
						APIServerID:     "1",
						EncodingVersion: "v1alpha1",
					},
					{
						APIServerID:     "2",
						EncodingVersion: "v1alpha1",
					},
				},
				CommonEncodingVersion: func() *string { a := "v1alpha1"; return &a }(),
			},
			expectedErr: "",
		},
		{
			status: apiserverinternal.StorageVersionStatus{
				StorageVersions: []apiserverinternal.ServerStorageVersion{
					{
						APIServerID:     "1",
						EncodingVersion: "v1alpha1",
					},
				},
				CommonEncodingVersion: func() *string { a := "v1alpha1"; return &a }(),
			},
			expectedErr: "",
		},
	}
	for _, tc := range cases {
		err := validateCommonVersion(tc.status, field.NewPath(""))
		if err == nil && len(tc.expectedErr) == 0 {
			continue
		}
		if err != nil && len(tc.expectedErr) == 0 {
			t.Errorf("unexpected error %v", err)
			continue
		}
		if err == nil && len(tc.expectedErr) != 0 {
			t.Errorf("unexpected empty error")
			continue
		}
		if !strings.Contains(err.Error(), tc.expectedErr) {
			t.Errorf("expected error to contain %s, got %s", tc.expectedErr, err)
		}
	}
}

func TestValidateStorageVersionCondition(t *testing.T) {
	cases := []struct {
		conditions  []apiserverinternal.StorageVersionCondition
		expectedErr string
	}{
		{
			conditions: []apiserverinternal.StorageVersionCondition{
				{
					Type:    "-fea",
					Status:  "True",
					Reason:  "unknown",
					Message: "unknown",
				},
			},
			expectedErr: "type: Invalid value",
		},
		{
			conditions: []apiserverinternal.StorageVersionCondition{
				{
					Type:    "fea",
					Status:  "-True",
					Reason:  "unknown",
					Message: "unknown",
				},
			},
			expectedErr: "status: Invalid value",
		},
		{
			conditions: []apiserverinternal.StorageVersionCondition{
				{
					Type:    "fea",
					Status:  "True",
					Message: "unknown",
				},
			},
			expectedErr: "Required value: reason cannot be empty",
		},
		{
			conditions: []apiserverinternal.StorageVersionCondition{
				{
					Type:   "fea",
					Status: "True",
					Reason: "unknown",
				},
			},
			expectedErr: "Required value: message cannot be empty",
		},
		{
			conditions: []apiserverinternal.StorageVersionCondition{
				{
					Type:    "fea",
					Status:  "True",
					Reason:  "unknown",
					Message: "unknown",
				},
				{
					Type:    "fea",
					Status:  "True",
					Reason:  "unknown",
					Message: "unknown",
				},
			},
			expectedErr: `"fea": the type of the condition is not unique, it also appears in conditions[0]`,
		},
		{
			conditions: []apiserverinternal.StorageVersionCondition{
				{
					Type:    "fea",
					Status:  "True",
					Reason:  "unknown",
					Message: "unknown",
				},
			},
			expectedErr: "",
		},
	}
	for _, tc := range cases {
		err := validateStorageVersionCondition(tc.conditions, field.NewPath("")).ToAggregate()
		if err == nil && len(tc.expectedErr) == 0 {
			continue
		}
		if err != nil && len(tc.expectedErr) == 0 {
			t.Errorf("unexpected error %v", err)
			continue
		}
		if err == nil && len(tc.expectedErr) != 0 {
			t.Errorf("unexpected empty error")
			continue
		}
		if !strings.Contains(err.Error(), tc.expectedErr) {
			t.Errorf("expected error to contain %s, got %s", tc.expectedErr, err)
		}
	}
}
