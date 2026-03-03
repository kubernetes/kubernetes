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
	"testing"

	"github.com/stretchr/testify/assert"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
)

func TestPodGroupStatusEqual(t *testing.T) {
	a := schedulingapi.PodGroupResourceClaimStatus{
		Name:              "a",
		ResourceClaimName: new("x"),
	}
	ay := schedulingapi.PodGroupResourceClaimStatus{
		Name:              "a",
		ResourceClaimName: new("y"),
	}
	b := schedulingapi.PodGroupResourceClaimStatus{
		Name:              "b",
		ResourceClaimName: new("y"),
	}
	empty := schedulingapi.PodGroupResourceClaimStatus{}

	testcases := map[string]struct {
		sliceA, sliceB []schedulingapi.PodGroupResourceClaimStatus
		expectEqual    bool
	}{
		"identical": {
			sliceA:      []schedulingapi.PodGroupResourceClaimStatus{a, b, empty},
			sliceB:      []schedulingapi.PodGroupResourceClaimStatus{a, b, empty},
			expectEqual: true,
		},
		"different-order": {
			sliceA:      []schedulingapi.PodGroupResourceClaimStatus{a, b, empty},
			sliceB:      []schedulingapi.PodGroupResourceClaimStatus{empty, a, b},
			expectEqual: false,
		},
		"different-length": {
			sliceA:      []schedulingapi.PodGroupResourceClaimStatus{a, b, empty},
			sliceB:      []schedulingapi.PodGroupResourceClaimStatus{a, b},
			expectEqual: false,
		},
		"different-resource-claim-name": {
			sliceA:      []schedulingapi.PodGroupResourceClaimStatus{a},
			sliceB:      []schedulingapi.PodGroupResourceClaimStatus{ay},
			expectEqual: false,
		},
		"semantically-equal-empty": {
			sliceA:      []schedulingapi.PodGroupResourceClaimStatus{},
			sliceB:      nil,
			expectEqual: true,
		},
		"semantically-equivalent": {
			sliceA:      []schedulingapi.PodGroupResourceClaimStatus{{Name: "a", ResourceClaimName: new("x")}},
			sliceB:      []schedulingapi.PodGroupResourceClaimStatus{{Name: "a", ResourceClaimName: new("x")}},
			expectEqual: true,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			assert.True(t, PodGroupStatusEqual(tc.sliceA, tc.sliceA), "%v", tc.sliceA)
			assert.True(t, PodGroupStatusEqual(tc.sliceB, tc.sliceB), "%v", tc.sliceB)
			assert.Equal(t, tc.expectEqual, PodGroupStatusEqual(tc.sliceA, tc.sliceB), "%v and %v", tc.sliceA, tc.sliceB)
		})

	}
}
