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

package resourceclaim

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
)

// TestExtendedResourceClaimName pins the derivation. The scheduler picks the
// name and the apiserver recomputes it to validate pod status. During an
// upgrade they run different versions, so if an expected value no longer
// matches, the change breaks scheduling of pods with extended resources backed
// by DRA on nodes where the devices consume node allocatable resources. Do not
// update the expected values.
func TestExtendedResourceClaimName(t *testing.T) {
	const (
		uid1 = types.UID("0e8b5a3c-6f1d-4b2a-9c7e-5d4f3a2b1c0d")
		uid2 = types.UID("8f2d6c1a-3e5b-4a7d-b9c0-1e2f3a4b5c6d")
	)
	tests := []struct {
		name    string
		podName string
		podUID  types.UID
		want    string
	}{
		{
			name:    "short pod name",
			podName: "my-pod",
			podUID:  uid1,
			want:    "my-pod-extended-resources-1468ca9b21",
		},
		{
			name:    "same pod name, different UID",
			podName: "my-pod",
			podUID:  uid2,
			want:    "my-pod-extended-resources-1d53d5de09",
		},
		{
			name:    "longest pod name without truncation",
			podName: strings.Repeat("a", 33),
			podUID:  uid1,
			want:    strings.Repeat("a", 33) + "-extended-resources-1468ca9b21",
		},
		{
			name:    "base truncated by one character",
			podName: strings.Repeat("a", 34),
			podUID:  uid1,
			want:    strings.Repeat("a", 34) + "-extended-resources1468ca9b21",
		},
		{
			name:    "longest pod name",
			podName: strings.Repeat("a", 253),
			podUID:  uid1,
			want:    strings.Repeat("a", 53) + "1468ca9b21",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ExtendedResourceClaimName(tc.podName, tc.podUID)
			if got != tc.want {
				t.Errorf("ExtendedResourceClaimName(%q, %q) = %q, want %q", tc.podName, tc.podUID, got, tc.want)
			}
			if len(got) > extendedResourceClaimNameMaxLength {
				t.Errorf("claim name %q has length %d, want at most %d", got, len(got), extendedResourceClaimNameMaxLength)
			}
			for _, msg := range validation.IsDNS1123Subdomain(got) {
				t.Errorf("claim name %q is not a valid ResourceClaim name: %s", got, msg)
			}
		})
	}
}
