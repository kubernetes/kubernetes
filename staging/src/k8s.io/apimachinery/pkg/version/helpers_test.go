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

package version

import (
	"testing"
)

func TestCompareKubeAwareVersionStrings(t *testing.T) {
	tests := []*struct {
		v1, v2          string
		expectedGreater bool
	}{
		{"v1", "v2", false},
		{"v2", "v1", true},
		{"v10", "v2", true},
		{"v1", "v2alpha1", true},
		{"v1", "v2beta1", true},
		{"v1alpha2", "v1alpha1", true},
		{"v1beta1", "v2alpha3", true},
		{"v1alpha10", "v1alpha2", true},
		{"v1beta10", "v1beta2", true},
		{"foo", "v1beta2", false},
		{"bar", "foo", true},
		{"version1", "version2", true},  // Non kube-like versions are sorted alphabetically
		{"version1", "version10", true}, // Non kube-like versions are sorted alphabetically
	}

	for _, tc := range tests {
		if e, a := tc.expectedGreater, CompareKubeAwareVersionStrings(tc.v1, tc.v2) > 0; e != a {
			if e {
				t.Errorf("expected %s to be greater than %s", tc.v1, tc.v2)
			} else {
				t.Errorf("expected %s to be less than %s", tc.v1, tc.v2)
			}
		}
	}
}

func Test_parseKubeVersion(t *testing.T) {
	tests := []struct {
		name             string
		v                string
		wantMajorVersion int
		wantVType        versionType
		wantMinorVersion int
		wantOk           bool
	}{
		{
			name:             "invaild version for ga",
			v:                "v1.1",
			wantMajorVersion: 0,
			wantVType:        0,
			wantMinorVersion: 0,
			wantOk:           false,
		},
		{
			name:             "invaild version for alpha",
			v:                "v1alpha1.1",
			wantMajorVersion: 0,
			wantVType:        0,
			wantMinorVersion: 0,
			wantOk:           false,
		},
		{
			name:             "alpha version",
			v:                "v1alpha1",
			wantMajorVersion: 1,
			wantVType:        0,
			wantMinorVersion: 1,
			wantOk:           true,
		},
		{
			name:             "beta version",
			v:                "v2beta10",
			wantMajorVersion: 2,
			wantVType:        1,
			wantMinorVersion: 10,
			wantOk:           true,
		},
		{
			name:             "ga version",
			v:                "v3",
			wantMajorVersion: 3,
			wantVType:        2,
			wantMinorVersion: 0,
			wantOk:           true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMajorVersion, gotVType, gotMinorVersion, gotOk := parseKubeVersion(tt.v)
			if gotMajorVersion != tt.wantMajorVersion {
				t.Errorf("parseKubeVersion() gotMajorVersion = %v, want %v", gotMajorVersion, tt.wantMajorVersion)
			}
			if gotVType != tt.wantVType {
				t.Errorf("parseKubeVersion() gotVType = %v, want %v", gotVType, tt.wantVType)
			}
			if gotMinorVersion != tt.wantMinorVersion {
				t.Errorf("parseKubeVersion() gotMinorVersion = %v, want %v", gotMinorVersion, tt.wantMinorVersion)
			}
			if gotOk != tt.wantOk {
				t.Errorf("parseKubeVersion() gotOk = %v, want %v", gotOk, tt.wantOk)
			}
		})
	}
}
