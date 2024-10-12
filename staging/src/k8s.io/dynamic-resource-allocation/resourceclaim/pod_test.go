/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestPodStatusEqual(t *testing.T) {
	a := corev1.PodResourceClaimStatus{
		Name:              "a",
		ResourceClaimName: ptr.To("x"),
	}
	b := corev1.PodResourceClaimStatus{
		Name:              "b",
		ResourceClaimName: ptr.To("y"),
	}
	empty := corev1.PodResourceClaimStatus{}

	testcases := map[string]struct {
		sliceA, sliceB []corev1.PodResourceClaimStatus
		expectEqual    bool
	}{
		"identical": {
			sliceA:      []corev1.PodResourceClaimStatus{a, b, empty},
			sliceB:      []corev1.PodResourceClaimStatus{a, b, empty},
			expectEqual: true,
		},
		"different-order": {
			sliceA:      []corev1.PodResourceClaimStatus{a, b, empty},
			sliceB:      []corev1.PodResourceClaimStatus{empty, a, b},
			expectEqual: false,
		},
		"different-length": {
			sliceA:      []corev1.PodResourceClaimStatus{a, b, empty},
			sliceB:      []corev1.PodResourceClaimStatus{a, b},
			expectEqual: false,
		},
		"semantically-equal-empty": {
			sliceA:      []corev1.PodResourceClaimStatus{},
			sliceB:      nil,
			expectEqual: true,
		},
		"semantically-equivalent": {
			sliceA:      []corev1.PodResourceClaimStatus{{Name: "a", ResourceClaimName: ptr.To("x")}},
			sliceB:      []corev1.PodResourceClaimStatus{{Name: "a", ResourceClaimName: ptr.To("x")}},
			expectEqual: true,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			assert.True(t, PodStatusEqual(tc.sliceA, tc.sliceA), fmt.Sprintf("%v", tc.sliceA))
			assert.True(t, PodStatusEqual(tc.sliceB, tc.sliceB), fmt.Sprintf("%v", tc.sliceB))
			assert.Equal(t, tc.expectEqual, PodStatusEqual(tc.sliceA, tc.sliceB), fmt.Sprintf("%v and %v", tc.sliceA, tc.sliceB))
		})

	}
}
