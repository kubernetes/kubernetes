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
	"reflect"
	"testing"

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
		PodGroupTemplates: []scheduling.PodGroupTemplate{
			{
				Name: "bar",
				SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
					Gang: &scheduling.GangSchedulingPolicy{
						MinCount: 5,
					},
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
		APIVersion:        "v1alpha2",
		Resource:          "workloads",
		IsResourceRequest: true,
	})
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
		workload.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = -1

		Strategy.PrepareForCreate(ctx, workload)
		errs := Strategy.Validate(ctx, workload)
		if len(errs) == 0 {
			t.Errorf("Expected validation error")
		}
	})
}

func TestWorkloadStrategyCreate(t *testing.T) {
	testCases := []struct {
		name                          string
		prepareWorkload               func(*scheduling.Workload)
		enableWorkloadAwarePreemption bool
		expectPriorityClassName       *string
		expectPriority                *int32
		expectValidationErrors        bool
	}{
		{
			name:                          "simple",
			prepareWorkload:               func(w *scheduling.Workload) {},
			enableWorkloadAwarePreemption: false,
			expectPriorityClassName:       nil,
			expectPriority:                nil,
			expectValidationErrors:        false,
		},
		{
			name:                          "failed validation",
			prepareWorkload:               func(w *scheduling.Workload) { w.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = -1 },
			enableWorkloadAwarePreemption: false,
			expectValidationErrors:        true,
		},
		{
			name:                          "priorityClassName is preserved upon creating workload when workload-aware-preemption is enabled",
			prepareWorkload:               func(w *scheduling.Workload) { w.Spec.PodGroupTemplates[0].PriorityClassName = ptr.To("high-priority") },
			enableWorkloadAwarePreemption: true,
			expectPriorityClassName:       ptr.To("high-priority"),
			expectPriority:                nil,
			expectValidationErrors:        false,
		},
		{
			name: "priorityClassName and priority are cleared upon creating workload when workload-aware-preemption is disabled",
			prepareWorkload: func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[0].PriorityClassName = ptr.To("high-priority")
				w.Spec.PodGroupTemplates[0].Priority = ptr.To(int32(1000))
			},
			enableWorkloadAwarePreemption: false,
			expectPriorityClassName:       nil,
			expectPriority:                nil,
			expectValidationErrors:        false,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:         true,
				features.GangScheduling:          tt.enableWorkloadAwarePreemption,
				features.WorkloadAwarePreemption: tt.enableWorkloadAwarePreemption,
			})

			ctx := ctxWithRequestInfo()
			w := workload.DeepCopy()
			tt.prepareWorkload(w)

			Strategy.PrepareForCreate(ctx, w)
			errs := Strategy.Validate(ctx, w)

			if (len(errs) != 0) != tt.expectValidationErrors {
				t.Errorf("Expected validation error = %v, got %v", tt.expectValidationErrors, errs)
			}

			if !reflect.DeepEqual(tt.expectPriorityClassName, w.Spec.PodGroupTemplates[0].PriorityClassName) {
				t.Errorf("Expected priorityClassName = %v, got %v", tt.expectPriorityClassName, w.Spec.PodGroupTemplates[0].PriorityClassName)
			}

			if !reflect.DeepEqual(tt.expectPriority, w.Spec.PodGroupTemplates[0].Priority) {
				t.Errorf("Expected priority = %v, got %v", tt.expectPriority, w.Spec.PodGroupTemplates[0].Priority)
			}
		})
	}
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

	t.Run("invalid spec update - controllerRef", func(t *testing.T) {
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
		if len(errs) == 0 {
			t.Errorf("Expected validation error")
		}
	})

	t.Run("invalid spec update - podGroupTemplates", func(t *testing.T) {
		ctx := ctxWithRequestInfo()
		workload := workload.DeepCopy()
		newWorkload := workload.DeepCopy()
		newWorkload.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = 4
		newWorkload.ResourceVersion = "4"
		Strategy.PrepareForUpdate(ctx, newWorkload, workload)
		errs := Strategy.ValidateUpdate(ctx, newWorkload, workload)
		if len(errs) == 0 {
			t.Errorf("Expected validation error")
		}
	})
}

func TestWorkloadStrategyUpdate(t *testing.T) {
	testCases := []struct {
		name                   string
		prepareOldWorkload     func(*scheduling.Workload)
		prepareNewWorkload     func(*scheduling.Workload)
		expectPriority         *int32
		expectValidationErrors bool
	}{
		{
			name:                   "no changes",
			prepareOldWorkload:     func(w *scheduling.Workload) {},
			prepareNewWorkload:     func(w *scheduling.Workload) {},
			expectValidationErrors: false,
		},
		{
			name:                   "name update",
			prepareOldWorkload:     func(w *scheduling.Workload) {},
			prepareNewWorkload:     func(w *scheduling.Workload) { w.Name += "bar" },
			expectValidationErrors: true,
		},
		{
			name:                   "spec update",
			prepareOldWorkload:     func(w *scheduling.Workload) {},
			prepareNewWorkload:     func(w *scheduling.Workload) { w.ObjectMeta.Labels = map[string]string{"test-key": "test-value"} },
			expectValidationErrors: false,
		},
		{
			name:                   "invalid spec update",
			prepareOldWorkload:     func(w *scheduling.Workload) {},
			prepareNewWorkload:     func(w *scheduling.Workload) { w.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = 4 },
			expectValidationErrors: true,
		},
		{
			name:                   "priorityClassName is immutable",
			prepareOldWorkload:     func(w *scheduling.Workload) { w.Spec.PodGroupTemplates[0].PriorityClassName = ptr.To("high-priority") },
			prepareNewWorkload:     func(w *scheduling.Workload) { w.Spec.PodGroupTemplates[0].PriorityClassName = ptr.To("low-priority") },
			expectValidationErrors: true,
		},
		{
			name:                   "priority is immutable",
			prepareOldWorkload:     func(w *scheduling.Workload) { w.Spec.PodGroupTemplates[0].Priority = ptr.To(int32(1000)) },
			prepareNewWorkload:     func(w *scheduling.Workload) { w.Spec.PodGroupTemplates[0].Priority = ptr.To(int32(2000)) },
			expectValidationErrors: true,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
				features.GangScheduling:  true,
			})

			ctx := ctxWithRequestInfo()
			old := workload.DeepCopy()
			new := workload.DeepCopy()
			new.ResourceVersion = "4"
			tt.prepareOldWorkload(old)
			tt.prepareNewWorkload(new)

			Strategy.PrepareForUpdate(ctx, new, old)
			errs := Strategy.ValidateUpdate(ctx, new, old)

			if (len(errs) != 0) != tt.expectValidationErrors {
				t.Errorf("Expected validation error = %v, got %v", tt.expectValidationErrors, errs)
			}

			if !tt.expectValidationErrors && !reflect.DeepEqual(tt.expectPriority, new.Spec.PodGroupTemplates[0].Priority) {
				t.Errorf("Expected priority = %v, got %v", tt.expectPriority, new.Spec.PodGroupTemplates[0].Priority)
			}
		})
	}
}
