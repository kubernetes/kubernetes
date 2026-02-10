/*
Copyright 2025 The Kubernetes Authors.

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

package workload

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

var workload = &scheduling.Workload{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "foo",
		Namespace: metav1.NamespaceDefault,
	},
	Spec: scheduling.WorkloadSpec{
		PodGroups: []scheduling.PodGroup{
			{
				Name: "bar",
				Policy: scheduling.PodGroupPolicy{
					Gang: &scheduling.GangSchedulingPolicy{
						MinCount: 5,
					},
				},
			},
			{
				Name: "baz",
				Policy: scheduling.PodGroupPolicy{
					Basic: &scheduling.BasicSchedulingPolicy{},
				},
			},
		},
	},
}

func TestWorkloadStrategy(t *testing.T) {
	if !Strategy.NamespaceScoped() {
		t.Errorf("Workload must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Workload should not allow create on update")
	}
}

func ctxWithRequestInfo() context.Context {
	return genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "workloads",
		IsResourceRequest: true,
	})
}

func TestWorkloadStrategyCreate(t *testing.T) {
	testCases := []struct {
		name                       string
		prepareWorkload            func(*scheduling.Workload)
		enablePodGroupDesiredCount bool
		expectedDesiredCounts      []*int32
		expectValidationErrors     bool
	}{
		{
			name:                   "simple",
			prepareWorkload:        func(w *scheduling.Workload) {},
			expectValidationErrors: false,
		},
		{
			name:                   "failed validation",
			prepareWorkload:        func(w *scheduling.Workload) { w.Spec.PodGroups[0].Policy.Gang.MinCount = -1 },
			expectValidationErrors: true,
		},
		{
			name: "PodGroupDesiredCount: disabled",
			prepareWorkload: func(w *scheduling.Workload) {
				w.Spec.PodGroups[0].Policy.Gang.DesiredCount = ptr.To(int32(6))
				w.Spec.PodGroups[1].Policy.Basic.DesiredCount = ptr.To(int32(6))
			},
			expectedDesiredCounts:  []*int32{nil, nil},
			expectValidationErrors: false,
		},
		{
			name: "PodGroupDesiredCount: enabled",
			prepareWorkload: func(w *scheduling.Workload) {
				w.Spec.PodGroups[0].Policy.Gang.DesiredCount = ptr.To(int32(6))
				w.Spec.PodGroups[1].Policy.Basic.DesiredCount = ptr.To(int32(6))
			},
			expectedDesiredCounts:      []*int32{ptr.To(int32(6)), ptr.To(int32(6))},
			enablePodGroupDesiredCount: true,
			expectValidationErrors:     false,
		},
		{
			name: "failed validation: desiredCount < minCount",
			prepareWorkload: func(w *scheduling.Workload) {
				w.Spec.PodGroups[0].Policy.Gang.DesiredCount = ptr.To(int32(4))
			},
			enablePodGroupDesiredCount: true,
			expectValidationErrors:     true,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:      true,
				features.PodGroupDesiredCount: tt.enablePodGroupDesiredCount,
			})

			ctx := ctxWithRequestInfo()
			w := workload.DeepCopy()
			tt.prepareWorkload(w)

			Strategy.PrepareForCreate(ctx, w)
			errs := Strategy.Validate(ctx, w)

			if (len(errs) != 0) != tt.expectValidationErrors {
				t.Errorf("Expected validation error = %v, got %v", tt.expectValidationErrors, errs)
			}

			if tt.expectedDesiredCounts != nil {
				for i, pg := range w.Spec.PodGroups {
					var actual *int32
					if pg.Policy.Gang != nil {
						actual = pg.Policy.Gang.DesiredCount
					} else if pg.Policy.Basic != nil {
						actual = pg.Policy.Basic.DesiredCount
					}
					if diff := cmp.Diff(tt.expectedDesiredCounts[i], actual); diff != "" {
						t.Errorf("PodGroup %d DesiredCount mismatch (-want +got):\n%s", i, diff)
					}
				}
			}
		})
	}
}

func TestPodSchedulingStrategyCreate(t *testing.T) {
	t.Run("simple", func(t *testing.T) {
		ctx := ctxWithRequestInfo()
		workload := workload.DeepCopy()

		Strategy.PrepareForCreate(ctx, workload)
		errs := Strategy.Validate(ctx, workload)
		if len(errs) != 0 {
			t.Errorf("Unexpected validation error: %v", errs)
		}
	})

	t.Run("failed validation", func(t *testing.T) {
		ctx := ctxWithRequestInfo()
		workload := workload.DeepCopy()
		workload.Spec.PodGroups[0].Policy.Gang.MinCount = -1

		Strategy.PrepareForCreate(ctx, workload)
		errs := Strategy.Validate(ctx, workload)
		if len(errs) == 0 {
			t.Errorf("Expected validation error")
		}
	})
}

func TestPodSchedulingStrategyUpdate(t *testing.T) {
	t.Run("no changes", func(t *testing.T) {
		ctx := ctxWithRequestInfo()
		workload := workload.DeepCopy()
		newWorkload := workload.DeepCopy()
		newWorkload.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newWorkload, workload)
		errs := Strategy.ValidateUpdate(ctx, newWorkload, workload)
		if len(errs) != 0 {
			t.Errorf("Unexpected validation error: %v", errs)
		}
	})

	t.Run("name update", func(t *testing.T) {
		ctx := ctxWithRequestInfo()
		workload := workload.DeepCopy()
		newWorkload := workload.DeepCopy()
		newWorkload.Name += "bar"
		newWorkload.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newWorkload, workload)
		errs := Strategy.ValidateUpdate(ctx, newWorkload, workload)
		if len(errs) == 0 {
			t.Errorf("Expected validation error")
		}
	})

	t.Run("spec update", func(t *testing.T) {
		ctx := ctxWithRequestInfo()
		workload := workload.DeepCopy()
		newWorkload := workload.DeepCopy()
		newWorkload.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
			Kind: "foo",
			Name: "baz",
		}
		newWorkload.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newWorkload, workload)
		errs := Strategy.ValidateUpdate(ctx, newWorkload, workload)
		if len(errs) != 0 {
			t.Errorf("Unexpected validation error: %v", errs)
		}
	})

	t.Run("invalid spec update", func(t *testing.T) {
		ctx := ctxWithRequestInfo()
		workload := workload.DeepCopy()
		newWorkload := workload.DeepCopy()
		newWorkload.Spec.PodGroups[0].Policy.Gang.MinCount = 4
		newWorkload.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newWorkload, workload)
		errs := Strategy.ValidateUpdate(ctx, newWorkload, workload)
		if len(errs) == 0 {
			t.Errorf("Expected validation error")
		}
	})
}
