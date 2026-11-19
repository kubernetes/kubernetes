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
	"context"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

const testPartitionAttr = "gpu.example.com/profile"

func testRequest(attr *string) *resource.ResourcePoolStatusRequest {
	return &resource.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "test-request", ResourceVersion: "1"},
		Spec: resource.ResourcePoolStatusRequestSpec{
			Driver:                        "gpu.example.com",
			Limit:                         ptr.To[int32](100),
			DefaultPartitionTypeAttribute: attr,
		},
	}
}

func testStatus() *resource.ResourcePoolStatusRequestStatus {
	return &resource.ResourcePoolStatusRequestStatus{
		PoolCount: ptr.To[int32](0),
		Conditions: []metav1.Condition{{
			Type:               resource.ResourcePoolStatusRequestConditionComplete,
			Status:             metav1.ConditionTrue,
			Reason:             "CalculationComplete",
			Message:            "Calculated status for 0 pools",
			LastTransitionTime: metav1.Now(),
		}},
	}
}

func TestStrategy(t *testing.T) {
	if Strategy.NamespaceScoped() {
		t.Error("ResourcePoolStatusRequest must be cluster scoped")
	}
	if Strategy.AllowCreateOnUpdate(context.Background()) {
		t.Error("ResourcePoolStatusRequest must not allow create on update")
	}
	if Strategy.AllowUnconditionalUpdate(context.Background()) {
		t.Error("ResourcePoolStatusRequest must not allow updates without a resourceVersion")
	}
}

// Status is ignored outside the status subresource.
func TestStrategyIgnoresStatus(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	request := testRequest(nil)
	request.Status = testStatus()
	Strategy.PrepareForCreate(ctx, request)
	if request.Status != nil {
		t.Errorf("PrepareForCreate kept status %+v, want nil", request.Status)
	}
	if errs := Strategy.Validate(ctx, request); len(errs) != 0 {
		t.Errorf("Validate: unexpected errors %v", errs)
	}

	oldRequest := testRequest(nil)
	oldRequest.Status = testStatus()
	newRequest := testRequest(nil)
	newRequest.Labels = map[string]string{"foo": "bar"}
	newRequest.Status = nil
	Strategy.PrepareForUpdate(ctx, newRequest, oldRequest)
	if diff := cmp.Diff(oldRequest.Status, newRequest.Status); diff != "" {
		t.Errorf("PrepareForUpdate changed the status (-old +new):\n%s", diff)
	}
	if errs := Strategy.ValidateUpdate(ctx, newRequest, oldRequest); len(errs) != 0 {
		t.Errorf("ValidateUpdate: unexpected errors %v", errs)
	}
}

func TestStatusStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testCases := map[string]struct {
		oldStatus *resource.ResourcePoolStatusRequestStatus
		update    func(*resource.ResourcePoolStatusRequest)
		wantErr   string
	}{
		"first write": {
			update: func(r *resource.ResourcePoolStatusRequest) { r.Status = testStatus() },
		},
		// Only the status changes.
		"spec and metadata changes are dropped": {
			update: func(r *resource.ResourcePoolStatusRequest) {
				r.Status = testStatus()
				r.Spec.Driver = "other.example.com"
				r.Labels = map[string]string{"foo": "bar"}
				r.Annotations = map[string]string{"foo": "bar"}
				r.Finalizers = []string{"foo"}
			},
		},
		"rename is rejected": {
			update: func(r *resource.ResourcePoolStatusRequest) {
				r.Status = testStatus()
				r.Name += "-2"
			},
			wantErr: "field is immutable",
		},
		// The status is write-once.
		"rewrite of a written status is rejected": {
			oldStatus: testStatus(),
			update:    func(r *resource.ResourcePoolStatusRequest) { r.Status.PoolCount = ptr.To[int32](1) },
			wantErr:   "field is immutable",
		},
		"clearing a written status is rejected": {
			oldStatus: testStatus(),
			update:    func(r *resource.ResourcePoolStatusRequest) { r.Status = nil },
			wantErr:   "field is immutable",
		},
		"invalid condition is rejected": {
			update: func(r *resource.ResourcePoolStatusRequest) {
				r.Status = testStatus()
				r.Status.Conditions[0].Reason = ""
			},
			wantErr: "status.conditions[0].reason",
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			oldRequest := testRequest(nil)
			oldRequest.Status = tc.oldStatus
			newRequest := oldRequest.DeepCopy()
			tc.update(newRequest)
			wantStatus := newRequest.Status.DeepCopy()

			StatusStrategy.PrepareForUpdate(ctx, newRequest, oldRequest)
			errs := StatusStrategy.ValidateUpdate(ctx, newRequest, oldRequest)
			if tc.wantErr != "" {
				if len(errs) == 0 || !strings.Contains(errs.ToAggregate().Error(), tc.wantErr) {
					t.Fatalf("ValidateUpdate errors = %v, want one containing %q", errs, tc.wantErr)
				}
				return
			}
			if len(errs) != 0 {
				t.Fatalf("ValidateUpdate: unexpected errors %v", errs)
			}

			want := oldRequest.DeepCopy()
			want.Status = wantStatus
			if diff := cmp.Diff(want, newRequest); diff != "" {
				t.Errorf("Unexpected object after PrepareForUpdate (-want +got):\n%s", diff)
			}
		})
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

			got := request.Spec.DefaultPartitionTypeAttribute
			if tc.wantDropped {
				if got != nil {
					t.Errorf("DefaultPartitionTypeAttribute = %q, want nil (gate disabled)", *got)
				}
				return
			}
			if got == nil || *got != testPartitionAttr {
				t.Errorf("DefaultPartitionTypeAttribute = %v, want %q", got, testPartitionAttr)
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

	if newRequest.Spec.DefaultPartitionTypeAttribute == nil {
		t.Error("DefaultPartitionTypeAttribute was dropped, want preserved (already in use)")
	}
}
