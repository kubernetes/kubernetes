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
	"k8s.io/kubernetes/pkg/apis/apiserverinternal"
	"k8s.io/utils/pointer"
)

func TestValidateServerStorageVersion(t *testing.T) {
	cases := []struct {
		ssv         apiserverinternal.ServerStorageVersion
		expectedErr string
	}{{
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "-fea",
			EncodingVersion:   "v1alpha1",
			DecodableVersions: []string{"v1alpha1", "v1"},
			ServedVersions:    []string{"v1alpha1", "v1"},
		},
		expectedErr: "apiServerID: Invalid value",
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1alpha1",
			DecodableVersions: []string{"v1beta1", "v1"},
			ServedVersions:    []string{"v1beta1", "v1"},
		},
		expectedErr: "decodableVersions must include encodingVersion",
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1alpha1",
			DecodableVersions: []string{"v1alpha1", "v1", "-fea"},
			ServedVersions:    []string{"v1alpha1", "v1", "-fea"},
		},
		expectedErr: "decodableVersions[2]: Invalid value",
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1alpha1",
			DecodableVersions: []string{"v1alpha1", "v1"},
			ServedVersions:    []string{"v1alpha1", "v1"},
		},
		expectedErr: "",
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1alpha1",
			DecodableVersions: []string{"v1alpha1", "v1"},
			ServedVersions:    []string{"v1alpha1", "v1"},
		},
		expectedErr: "",
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "mygroup.com/v2",
			DecodableVersions: []string{"v1alpha1", "v1", "mygroup.com/v2"},
			ServedVersions:    []string{"v1alpha1", "v1", "mygroup.com/v2"},
		},
		expectedErr: "",
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1alpha1",
			DecodableVersions: []string{"v1alpha1", "v1"},
			ServedVersions:    []string{"/v3"},
		},
		expectedErr: `[].servedVersions[0]: Invalid value: "/v3": group part: must be non-empty`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "mygroup.com/v2",
			DecodableVersions: []string{"mygroup.com/v2", "/v3"},
			ServedVersions:    []string{"mygroup.com/v2", "/v3"},
		},
		expectedErr: `[].decodableVersions[1]: Invalid value: "/v3": group part: must be non-empty`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "mygroup.com/v2",
			DecodableVersions: []string{"mygroup.com/v2", "/v3"},
			ServedVersions:    []string{"mygroup.com/"},
		},
		expectedErr: `[].servedVersions[0]: Invalid value: "mygroup.com/": version part: must be non-empty`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "mygroup.com/v2",
			DecodableVersions: []string{"mygroup.com/v2", "mygroup.com/"},
			ServedVersions:    []string{"mygroup.com/v2", "mygroup.com/"},
		},
		expectedErr: `[].decodableVersions[1]: Invalid value: "mygroup.com/": version part: must be non-empty`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "/v3",
			DecodableVersions: []string{"mygroup.com/v2", "/v3"},
			ServedVersions:    []string{"mygroup.com/v2", "/v3"},
		},
		expectedErr: `[].encodingVersion: Invalid value: "/v3": group part: must be non-empty`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1",
			DecodableVersions: []string{"v1", "mygroup_com/v2"},
			ServedVersions:    []string{"v1", "mygroup_com/v2"},
		},
		expectedErr: `[].decodableVersions[1]: Invalid value: "mygroup_com/v2": group part: a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1",
			DecodableVersions: []string{"v1", "mygroup.com/v2"},
			ServedVersions:    []string{"v1", "mygroup_com/v2"},
		},
		expectedErr: `[].servedVersions[1]: Invalid value: "mygroup_com/v2": group part: a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1",
			DecodableVersions: []string{"v1", "mygroup.com/v2_"},
			ServedVersions:    []string{"v1", "mygroup.com/v2_"},
		},
		expectedErr: `[].decodableVersions[1]: Invalid value: "mygroup.com/v2_": version part: a DNS-1035 label must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123', regex used for validation is '[a-z]([-a-z0-9]*[a-z0-9])?')`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1",
			DecodableVersions: []string{"v1", "mygroup.com/v2"},
			ServedVersions:    []string{"v1", "mygroup.com/v2_"},
		},
		expectedErr: `[].servedVersions[1]: Invalid value: "mygroup.com/v2_": version part: a DNS-1035 label must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123', regex used for validation is '[a-z]([-a-z0-9]*[a-z0-9])?')`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1",
			DecodableVersions: []string{"v1", "mygroup.com/v2/myresource"},
			ServedVersions:    []string{"v1", "mygroup.com/v2/myresource"},
		},
		expectedErr: `[].decodableVersions[1]: Invalid value: "mygroup.com/v2/myresource": an apiVersion is a DNS-1035 label, which must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123', regex used for validation is '[a-z]([-a-z0-9]*[a-z0-9])?') with an optional DNS subdomain prefix and '/' (e.g. 'example.com/MyVersion')`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1",
			DecodableVersions: []string{"v1", "mygroup.com/v2"},
			ServedVersions:    []string{"v1", "mygroup.com/v2/myresource"},
		},
		expectedErr: `[].servedVersions[1]: Invalid value: "mygroup.com/v2/myresource": an apiVersion is a DNS-1035 label, which must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123', regex used for validation is '[a-z]([-a-z0-9]*[a-z0-9])?') with an optional DNS subdomain prefix and '/' (e.g. 'example.com/MyVersion')`,
	}, {
		ssv: apiserverinternal.ServerStorageVersion{
			APIServerID:       "fea",
			EncodingVersion:   "v1alpha1",
			DecodableVersions: []string{"v1alpha1", "v1"},
			ServedVersions:    []string{"v2"},
		},
		expectedErr: `[].servedVersions[0]: Invalid value: "v2": individual served version : v2 must be included in decodableVersions : [v1alpha1 v1]`,
	}}

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

func TestValidateStorageVersionStatus(t *testing.T) {
	cases := []struct {
		svs         apiserverinternal.StorageVersionStatus
		expectedErr string
	}{{
		svs: apiserverinternal.StorageVersionStatus{
			StorageVersions: []apiserverinternal.ServerStorageVersion{{
				APIServerID:       "1",
				EncodingVersion:   "v1alpha1",
				DecodableVersions: []string{"v1alpha1", "v1"},
			}, {
				APIServerID:       "2",
				EncodingVersion:   "v1alpha1",
				DecodableVersions: []string{"v1alpha1", "v1"},
			}},
			CommonEncodingVersion: pointer.String("v1alpha1"),
		},
		expectedErr: "",
	}, {
		svs: apiserverinternal.StorageVersionStatus{
			StorageVersions: []apiserverinternal.ServerStorageVersion{{
				APIServerID:       "1",
				EncodingVersion:   "v1alpha1",
				DecodableVersions: []string{"v1alpha1", "v1"},
			}, {
				APIServerID:       "1",
				EncodingVersion:   "v1beta1",
				DecodableVersions: []string{"v1alpha1", "v1"},
			}},
			CommonEncodingVersion: pointer.String("v1alpha1"),
		},
		expectedErr: "storageVersions[1].apiServerID: Duplicate value: \"1\"",
	}}

	for _, tc := range cases {
		err := validateStorageVersionStatus(tc.svs, field.NewPath("")).ToAggregate()
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
	}{{
		status: apiserverinternal.StorageVersionStatus{
			StorageVersions:       []apiserverinternal.ServerStorageVersion{},
			CommonEncodingVersion: func() *string { a := "v1alpha1"; return &a }(),
		},
		expectedErr: "should be nil if servers do not agree on the same encoding version, or if there is no server reporting the supported versions yet",
	}, {
		status: apiserverinternal.StorageVersionStatus{
			StorageVersions: []apiserverinternal.ServerStorageVersion{{
				APIServerID:     "1",
				EncodingVersion: "v1alpha1",
			}, {
				APIServerID:     "2",
				EncodingVersion: "v1",
			}},
			CommonEncodingVersion: func() *string { a := "v1alpha1"; return &a }(),
		},
		expectedErr: "should be nil if servers do not agree on the same encoding version, or if there is no server reporting the supported versions yet",
	}, {
		status: apiserverinternal.StorageVersionStatus{
			StorageVersions: []apiserverinternal.ServerStorageVersion{{
				APIServerID:     "1",
				EncodingVersion: "v1alpha1",
			}, {
				APIServerID:     "2",
				EncodingVersion: "v1alpha1",
			}},
			CommonEncodingVersion: nil,
		},
		expectedErr: "Invalid value: \"null\": the common encoding version is v1alpha1",
	}, {
		status: apiserverinternal.StorageVersionStatus{
			StorageVersions: []apiserverinternal.ServerStorageVersion{{
				APIServerID:     "1",
				EncodingVersion: "v1alpha1",
			}, {
				APIServerID:     "2",
				EncodingVersion: "v1alpha1",
			}},
			CommonEncodingVersion: func() *string { a := "v1"; return &a }(),
		},
		expectedErr: "Invalid value: \"v1\": the actual common encoding version is v1alpha1",
	}, {
		status: apiserverinternal.StorageVersionStatus{
			StorageVersions: []apiserverinternal.ServerStorageVersion{{
				APIServerID:     "1",
				EncodingVersion: "v1alpha1",
			}, {
				APIServerID:     "2",
				EncodingVersion: "v1alpha1",
			}},
			CommonEncodingVersion: func() *string { a := "v1alpha1"; return &a }(),
		},
		expectedErr: "",
	}, {
		status: apiserverinternal.StorageVersionStatus{
			StorageVersions: []apiserverinternal.ServerStorageVersion{{
				APIServerID:     "1",
				EncodingVersion: "v1alpha1",
			}},
			CommonEncodingVersion: func() *string { a := "v1alpha1"; return &a }(),
		},
		expectedErr: "",
	}}
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
	}{{
		conditions: []apiserverinternal.StorageVersionCondition{{
			Type:    "-fea",
			Status:  "True",
			Reason:  "unknown",
			Message: "unknown",
		}},
		expectedErr: "type: Invalid value",
	}, {
		conditions: []apiserverinternal.StorageVersionCondition{{
			Type:    "fea",
			Status:  "-True",
			Reason:  "unknown",
			Message: "unknown",
		}},
		expectedErr: "status: Invalid value",
	}, {
		conditions: []apiserverinternal.StorageVersionCondition{{
			Type:    "fea",
			Status:  "True",
			Message: "unknown",
		}},
		expectedErr: "Required value: reason cannot be empty",
	}, {
		conditions: []apiserverinternal.StorageVersionCondition{{
			Type:   "fea",
			Status: "True",
			Reason: "unknown",
		}},
		expectedErr: "Required value: message cannot be empty",
	}, {
		conditions: []apiserverinternal.StorageVersionCondition{{
			Type:    "fea",
			Status:  "True",
			Reason:  "unknown",
			Message: "unknown",
		}, {
			Type:    "fea",
			Status:  "True",
			Reason:  "unknown",
			Message: "unknown",
		}},
		expectedErr: `"fea": the type of the condition is not unique, it also appears in conditions[0]`,
	}, {
		conditions: []apiserverinternal.StorageVersionCondition{{
			Type:    "fea",
			Status:  "True",
			Reason:  "unknown",
			Message: "unknown",
		}},
		expectedErr: "",
	}}
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

func TestValidateStorageVersionName(t *testing.T) {
	cases := []struct {
		name        string
		expectedErr string
	}{{
		name:        "",
		expectedErr: `name must be in the form of <group>.<resource>`,
	}, {
		name:        "pods",
		expectedErr: `name must be in the form of <group>.<resource>`,
	}, {
		name:        "core.pods",
		expectedErr: "",
	}, {
		name:        "authentication.k8s.io.tokenreviews",
		expectedErr: "",
	}, {
		name:        strings.Repeat("x", 253) + ".tokenreviews",
		expectedErr: "",
	}, {
		name:        strings.Repeat("x", 254) + ".tokenreviews",
		expectedErr: `the group segment must be no more than 253 characters`,
	}, {
		name:        "authentication.k8s.io." + strings.Repeat("x", 63),
		expectedErr: "",
	}, {
		name:        "authentication.k8s.io." + strings.Repeat("x", 64),
		expectedErr: `the resource segment must be no more than 63 characters`,
	}}
	for _, tc := range cases {
		errs := ValidateStorageVersionName(tc.name, false)
		if errs == nil && len(tc.expectedErr) == 0 {
			continue
		}
		if errs != nil && len(tc.expectedErr) == 0 {
			t.Errorf("unexpected error %v", errs)
			continue
		}
		if errs == nil && len(tc.expectedErr) != 0 {
			t.Errorf("unexpected empty error")
			continue
		}
		found := false
		for _, msg := range errs {
			if msg == tc.expectedErr {
				found = true
			}
		}
		if !found {
			t.Errorf("expected error to contain %s, got %v", tc.expectedErr, errs)
		}
	}
}
