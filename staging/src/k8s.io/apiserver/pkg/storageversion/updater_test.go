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
)

func TestLocalUpdateStorageVersion(t *testing.T) {
	v1 := "v1"
	ssv1 := v1alpha1.ServerStorageVersion{
		APIServerID:       "1",
		EncodingVersion:   "v1",
		DecodableVersions: []string{"v1", "v2"},
	}
	ssv2 := v1alpha1.ServerStorageVersion{
		APIServerID:       "2",
		EncodingVersion:   "v1",
		DecodableVersions: []string{"v1", "v2"},
	}
	// ssv3 has a different encoding version
	ssv3 := v1alpha1.ServerStorageVersion{
		APIServerID:       "3",
		EncodingVersion:   "v2",
		DecodableVersions: []string{"v1", "v2"},
	}
	ssv4 := v1alpha1.ServerStorageVersion{
		APIServerID:       "4",
		EncodingVersion:   "v1",
		DecodableVersions: []string{"v1", "v2", "v4"},
	}
	tests := []struct {
		old      v1alpha1.StorageVersionStatus
		newSSV   v1alpha1.ServerStorageVersion
		expected v1alpha1.StorageVersionStatus
	}{
		{
			old:    v1alpha1.StorageVersionStatus{},
			newSSV: ssv1,
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1},
				CommonEncodingVersion: &v1,
			},
		},
		{
			old: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2},
				CommonEncodingVersion: &v1,
			},
			newSSV: ssv3,
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
			},
		},
		{
			old: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2},
				CommonEncodingVersion: &v1,
			},
			newSSV: ssv4,
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions:       []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv4},
				CommonEncodingVersion: &v1,
			},
		},
		{
			old: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3},
			},
			newSSV: ssv4,
			expected: v1alpha1.StorageVersionStatus{
				StorageVersions: []v1alpha1.ServerStorageVersion{ssv1, ssv2, ssv3, ssv4},
			},
		},
	}

	for _, tc := range tests {
		sv := &v1alpha1.StorageVersion{Status: tc.old}
		updated := localUpdateStorageVersion(sv, tc.newSSV.APIServerID, tc.newSSV.EncodingVersion, tc.newSSV.DecodableVersions)
		if e, a := tc.expected, updated.Status; !reflect.DeepEqual(e, a) {
			t.Errorf("unexpected: %v", cmp.Diff(e, a))
		}
	}
}
