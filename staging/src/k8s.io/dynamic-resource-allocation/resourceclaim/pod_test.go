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
	ay := corev1.PodResourceClaimStatus{
		Name:              "a",
		ResourceClaimName: ptr.To("y"),
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
		"different-resource-claim-name": {
			sliceA:      []corev1.PodResourceClaimStatus{a},
			sliceB:      []corev1.PodResourceClaimStatus{ay},
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

func TestPodExtendedStatusEqual(t *testing.T) {
	ra := corev1.ContainerExtendedResourceRequest{
		ContainerName: "a",
		ResourceName:  "example.com/a",
		RequestName:   "ra",
	}
	rb := corev1.ContainerExtendedResourceRequest{
		ContainerName: "b",
		ResourceName:  "example.com/b",
		RequestName:   "rb",
	}

	testcases := map[string]struct {
		statusA, statusB *corev1.PodExtendedResourceClaimStatus
		expectEqual      bool
	}{
		"identical": {
			statusA: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{ra, rb},
				ResourceClaimName: "x",
			},
			statusB: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{ra, rb},
				ResourceClaimName: "x",
			},
			expectEqual: true,
		},
		"different-order": {
			statusA: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{ra, rb},
				ResourceClaimName: "x",
			},
			statusB: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{rb, ra},
				ResourceClaimName: "x",
			},
			expectEqual: false,
		},
		"different-length": {
			statusA: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{ra, rb},
				ResourceClaimName: "x",
			},
			statusB: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{ra},
				ResourceClaimName: "x",
			},
			expectEqual: false,
		},

		"semantically-equal-empty": {
			statusA: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{},
				ResourceClaimName: "x",
			},
			statusB: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   nil,
				ResourceClaimName: "x",
			},
			expectEqual: true,
		},
		"different-claim-name": {
			statusA: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{ra, rb},
				ResourceClaimName: "x",
			},
			statusB: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{ra, rb},
				ResourceClaimName: "y",
			},
			expectEqual: false,
		},
		"both-nil": {
			statusA:     nil,
			statusB:     nil,
			expectEqual: true,
		},
		"one-nil-other-not-nil": {
			statusA: nil,
			statusB: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{ra, rb},
				ResourceClaimName: "y",
			},
			expectEqual: false,
		},
		"one-not-nil-other-nil": {
			statusA: &corev1.PodExtendedResourceClaimStatus{
				RequestMappings:   []corev1.ContainerExtendedResourceRequest{ra, rb},
				ResourceClaimName: "x",
			},
			statusB:     nil,
			expectEqual: false,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			assert.True(t, PodExtendedStatusEqual(tc.statusA, tc.statusA), "status", tc.statusA)
			assert.True(t, PodExtendedStatusEqual(tc.statusB, tc.statusB), "status", tc.statusB)
			assert.Equal(t, tc.expectEqual, PodExtendedStatusEqual(tc.statusA, tc.statusB), "status", tc.statusA, tc.statusB)
		})
	}
}
