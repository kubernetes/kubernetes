/*
Copyright 2019 The Kubernetes Authors.

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

package storageversion

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/api/apiserverinternal/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	v1   = "v1"
	v2   = "v2"
	ssv1 = v1alpha1.ServerStorageVersion{
		APIServerID:       "1",
		EncodingVersion:   "v1",
		DecodableVersions: []string{"v1", "v2"},
		ServedVersions:    []string{"v1"},
	}
	ssv2 = v1alpha1.ServerStorageVersion{
		APIServerID:       "2",
		EncodingVersion:   "v1",
		DecodableVersions: []string{"v1", "v2"},
		ServedVersions:    []string{"v1", "v2"},
	}
	// ssv3 has a different encoding version
	ssv3 = v1alpha1.ServerStorageVersion{
		APIServerID:       "3",
		EncodingVersion:   "v2",
		DecodableVersions: []string{"v1", "v2"},
		ServedVersions:    []string{"v1", "v2"},
	}
	ssv4 = v1alpha1.ServerStorageVersion{
		APIServerID:       "4",
		EncodingVersion:   "v1",
		DecodableVersions: []string{"v1", "v2", "v4"},
		ServedVersions:    []string{"v1", "v2"},
	}
	ssv5 = v1alpha1.ServerStorageVersion{
		APIServerID:       "5",
		EncodingVersion:   "v2",
		DecodableVersions: []string{"v1", "v2", "v4"},
		ServedVersions:    []string{"v1", "v2"},
	}
)

func TestLocalUpdateStorageVersion(t *testing.T) {
	tests := []struct {
		old                            v1alpha1.StorageVersionStatus
		newSSV                         v1alpha1.ServerStorageVersion
		expected                       v1alpha1.StorageVersionStatus
		expectLastTransitionTimeUpdate bool
	}{
		{
			old:    v1alpha1.StorageVersionStatus{},
			newSSV: ssv1,
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
			expectLastTransitionTimeUpdate: true,
		},
		{
			old: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
			newSSV: ssv3,
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
				Conditions:      commonVersionFalseCondition(),
			},
			expectLastTransitionTimeUpdate: true,
		},
		{
			old: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
			newSSV: ssv4,
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv4},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
		},
		{
			old: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
				Conditions:      commonVersionFalseCondition(),
			},
			newSSV: ssv4,
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3, ssv4},
				Conditions:      commonVersionFalseCondition(),
			},
		},
	}

	for _, tc := range tests {
		sv := &v1alpha1.StorageVersion{Status: tc.old}
		updated := localUpdateStorageVersion(sv, tc.newSSV.APIServerID, tc.newSSV.EncodingVersion, tc.newSSV.DecodableVersions, tc.newSSV.ServedVersions)
		if tc.expectLastTransitionTimeUpdate == updated.Status.Conditions[0].LastTransitionTime.IsZero() {
			t.Errorf("unexpected LastTransitionTime, expected update: %v, got: %v",
				tc.expectLastTransitionTimeUpdate, updated.Status.Conditions[0].LastTransitionTime)
		}
		updated.Status.Conditions[0].LastTransitionTime = metav1.Time{}
		if e, a := tc.expected, updated.Status; !reflect.DeepEqual(e, a) {
			t.Errorf("unexpected: %v", cmp.Diff(e, a))
		}
	}
}

func commonVersionTrueCondition() []v1alpha1.StorageVersionCondition {
	return []v1alpha1.StorageVersionCondition{{
		Type:    v1alpha1.AllEncodingVersionsEqual,
		Status:  v1alpha1.ConditionTrue,
		Reason:  "CommonEncodingVersionSet",
		Message: "Common encoding version set",
	}}
}
func commonVersionFalseCondition() []v1alpha1.StorageVersionCondition {
	return []v1alpha1.StorageVersionCondition{{
		Type:    v1alpha1.AllEncodingVersionsEqual,
		Status:  v1alpha1.ConditionFalse,
		Reason:  "CommonEncodingVersionUnset",
		Message: "Common encoding version unset",
	}}
}

func TestSetCommonEncodingVersion(t *testing.T) {
	tests := []struct {
		name                           string
		old                            v1alpha1.StorageVersionStatus
		expected                       v1alpha1.StorageVersionStatus
		expectLastTransitionTimeUpdate bool
	}{
		{
			name: "no-common_init",
			old: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
			},
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
				Conditions:      commonVersionFalseCondition(),
			},
			expectLastTransitionTimeUpdate: true,
		},
		{
			name: "no-common_transition",
			old: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
				Conditions:      commonVersionFalseCondition(),
			},
			expectLastTransitionTimeUpdate: true,
		},
		{
			name: "no-common_no-transition",
			old: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
				Conditions:      commonVersionFalseCondition(),
			},
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
				Conditions:      commonVersionFalseCondition(),
			},
		},
		{
			name: "common_init",
			old: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2},
			},
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
			expectLastTransitionTimeUpdate: true,
		},
		{
			name: "common_no-transition",
			old: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
		},
		{
			name: "common_transition",
			old: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2},
				Conditions:      commonVersionFalseCondition(),
			},
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
			expectLastTransitionTimeUpdate: true,
		},
		{
			name: "common_version-changed",
			old: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv3, ssv5},
				CommonEncodingVersion: &v1,
				Conditions:            commonVersionTrueCondition(),
			},
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv3, ssv5},
				CommonEncodingVersion: &v2,
				Conditions:            commonVersionTrueCondition(),
			},
			expectLastTransitionTimeUpdate: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sv := &v1alpha1.StorageVersion{Status: tc.old}
			SetCommonEncodingVersion(sv)
			if (tc.expectLastTransitionTimeUpdate && sv.Status.Conditions[0].LastTransitionTime.IsZero()) ||
				(!tc.expectLastTransitionTimeUpdate && !sv.Status.Conditions[0].LastTransitionTime.IsZero()) {
				t.Errorf("unexpected LastTransitionTime, expected update: %v, got: %v",
					tc.expectLastTransitionTimeUpdate, sv.Status.Conditions[0].LastTransitionTime)
			}
			sv.Status.Conditions[0].LastTransitionTime = metav1.Time{}
			if e, a := tc.expected, sv.Status; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected: %v", cmp.Diff(e, a))
			}
		})
	}
}
