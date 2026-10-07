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
)

func TestExtendedResourceClaimNameBase(t *testing.T) {
	tests := []struct {
		name    string
		podName string
		want    string
	}{
		{
			name:    "short pod name",
			podName: "my-pod",
			want:    "my-pod-extended-resources-",
		},
		{
			name:    "longest pod name without truncation",
			podName: strings.Repeat("a", 38),
			want:    strings.Repeat("a", 38) + "-extended-resources-",
		},
		{
			name:    "truncated suffix",
			podName: strings.Repeat("a", 39),
			want:    strings.Repeat("a", 39) + "-extended-resources",
		},
		{
			name:    "truncated pod name",
			podName: strings.Repeat("a", 100),
			want:    strings.Repeat("a", 58),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := ExtendedResourceClaimNameBase(tc.podName); got != tc.want {
				t.Errorf("ExtendedResourceClaimNameBase(%q) = %q, want %q", tc.podName, got, tc.want)
			}
		})
	}
}

func TestIsExtendedResourceClaimNameForPod(t *testing.T) {
	longPodName := strings.Repeat("a", 50)
	tests := []struct {
		name      string
		podName   string
		claimName string
		want      bool
	}{
		{
			name:      "generated for the pod",
			podName:   "my-pod",
			claimName: "my-pod-extended-resources-abcde",
			want:      true,
		},
		{
			name:      "generated for a long pod name",
			podName:   longPodName,
			claimName: longPodName + "-extende" + "abcde",
			want:      true,
		},
		{
			name:      "untruncated base for a long pod name",
			podName:   longPodName,
			claimName: longPodName + "-extended-resources-abcde",
			want:      true,
		},
		{
			name:      "another pod",
			podName:   "my-pod",
			claimName: "other-pod-extended-resources-abcde",
		},
		{
			name:      "not generated from the base",
			podName:   "my-pod",
			claimName: "my-pod-abcde",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsExtendedResourceClaimNameForPod(tc.podName, tc.claimName); got != tc.want {
				t.Errorf("IsExtendedResourceClaimNameForPod(%q, %q) = %v, want %v", tc.podName, tc.claimName, got, tc.want)
			}
		})
	}
}
