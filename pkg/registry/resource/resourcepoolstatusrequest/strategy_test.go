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

package resourcepoolstatusrequest

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
)

const testPartitionAttr = "gpu.example.com/profile"

func testRequest(attr *string) *resource.ResourcePoolStatusRequest {
	return &resource.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "test-request"},
		Spec: resource.ResourcePoolStatusRequestSpec{
			Driver:                 "gpu.example.com",
			PartitionTypeAttribute: attr,
		},
	}
}

// The partition type default is only accepted while DRAPartitionableDevicesType
// is enabled; otherwise it is dropped so the feature cannot be used piecemeal.
func TestPrepareForCreatePartitionTypeAttribute(t *testing.T) {
	testCases := map[string]struct {
		featureOverrides map[featuregate.Feature]bool
		wantDropped      bool
	}{
		"gate-enabled-keeps": {
			featureOverrides: map[featuregate.Feature]bool{
				features.DRAResourcePoolStatus:       true,
				features.DRAPartitionableDevicesType: true,
			},
		},
		"gate-disabled-drops": {
			featureOverrides: map[featuregate.Feature]bool{
				features.DRAResourcePoolStatus:       true,
				features.DRAPartitionableDevicesType: false,
			},
			wantDropped: true,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.featureOverrides)
			request := testRequest(new(testPartitionAttr))
			Strategy.PrepareForCreate(genericapirequest.NewDefaultContext(), request)

			got := request.Spec.PartitionTypeAttribute
			if tc.wantDropped {
				if got != nil {
					t.Errorf("PartitionTypeAttribute = %q, want nil (gate disabled)", *got)
				}
				return
			}
			if got == nil || *got != testPartitionAttr {
				t.Errorf("PartitionTypeAttribute = %v, want %q", got, testPartitionAttr)
			}
		})
	}
}

// A default already stored on the object survives a disabled gate.
func TestPrepareForUpdateKeepsExistingPartitionTypeAttribute(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, map[featuregate.Feature]bool{
		features.DRAResourcePoolStatus:       true,
		features.DRAPartitionableDevicesType: false,
	})
	oldRequest := testRequest(new(testPartitionAttr))
	newRequest := testRequest(new(testPartitionAttr))
	Strategy.PrepareForUpdate(genericapirequest.NewDefaultContext(), newRequest, oldRequest)

	if newRequest.Spec.PartitionTypeAttribute == nil {
		t.Error("PartitionTypeAttribute was dropped, want preserved (already in use)")
	}
}
