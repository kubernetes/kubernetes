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
		PodGroups: []scheduling.PodGroup{
			{
				Name: "bar",
				Policy: scheduling.PodGroupPolicy{
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

func TestWorkloadStrategyCreate(t *testing.T) {
	testCases := []struct {
		name                           string
		prepareWorkload                func(*scheduling.Workload)
		workloadAwarePreemptionEnabled bool
		expectPriorityClassName        *string
		expectPriority                 *int32
		expectValidationErrors         bool
	}{
		{
			name:                           "simple",
			prepareWorkload:                func(w *scheduling.Workload) {},
			workloadAwarePreemptionEnabled: false,
			expectPriorityClassName:        nil,
			expectPriority:                 nil,
			expectValidationErrors:         false,
		},
		{
			name:                           "failed validation",
			prepareWorkload:                func(w *scheduling.Workload) { w.Spec.PodGroups[0].Policy.Gang.MinCount = -1 },
			workloadAwarePreemptionEnabled: false,
			expectValidationErrors:         true,
		},
		{
			name:                           "priorityClassName is preserved upon creating workload when workload-aware-preemption is enabled",
			prepareWorkload:                func(w *scheduling.Workload) { w.Spec.PriorityClassName = ptr.To("high-priority") },
			workloadAwarePreemptionEnabled: true,
			expectPriorityClassName:        ptr.To("high-priority"),
			expectPriority:                 nil,
			expectValidationErrors:         false,
		},
		{
			name:                           "priorityClassName is cleared upon creating workload when workload-aware-preemption is disabled",
			prepareWorkload:                func(w *scheduling.Workload) { w.Spec.PriorityClassName = ptr.To("high-priority") },
			workloadAwarePreemptionEnabled: false,
			expectPriorityClassName:        nil,
			expectPriority:                 nil,
			expectValidationErrors:         false,
		},
		{
			name:                           "priority field is always cleared on create",
			prepareWorkload:                func(w *scheduling.Workload) { w.Spec.Priority = ptr.To(int32(1000)) },
			workloadAwarePreemptionEnabled: false,
			expectPriority:                 nil,
			expectValidationErrors:         false,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:         true,
				features.GangScheduling:          true,
				features.WorkloadAwarePreemption: tt.workloadAwarePreemptionEnabled,
			})

			ctx := genericapirequest.NewDefaultContext()
			w := workload.DeepCopy()
			tt.prepareWorkload(w)

			Strategy.PrepareForCreate(ctx, w)
			errs := Strategy.Validate(ctx, w)

			if (len(errs) != 0) != tt.expectValidationErrors {
				t.Errorf("Expected validation error = %v, got %v", tt.expectValidationErrors, errs)
			}

			if !reflect.DeepEqual(tt.expectPriorityClassName, w.Spec.PriorityClassName) {
				t.Errorf("Expected priorityClassName = %v, got %v", tt.expectPriorityClassName, w.Spec.PriorityClassName)
			}

			if tt.expectPriority != nil && !reflect.DeepEqual(tt.expectPriority, w.Spec.Priority) {
				t.Errorf("Expected priority = %v, got %v", tt.expectPriority, w.Spec.Priority)
			}
		})
	}
}

func TestWorkloadStrategyUpdate(t *testing.T) {

	testCases := []struct {
		name                   string
		prepareOldWorkload     func(*scheduling.Workload)
		prepareNewWorkload     func(*scheduling.Workload)
		expectPriority         int32
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
			prepareNewWorkload:     func(w *scheduling.Workload) { w.Spec.PodGroups[0].Policy.Gang.MinCount = 4 },
			expectValidationErrors: true,
		},
		{
			name:                   "priorityClassName is immutable",
			prepareOldWorkload:     func(w *scheduling.Workload) { w.Spec.PriorityClassName = ptr.To("high-priority") },
			prepareNewWorkload:     func(w *scheduling.Workload) { w.Spec.PriorityClassName = ptr.To("low-priority") },
			expectValidationErrors: true,
		},
		{
			name:                   "priority field is always preserved on update",
			prepareOldWorkload:     func(w *scheduling.Workload) { w.Spec.Priority = ptr.To(int32(1000)) },
			prepareNewWorkload:     func(w *scheduling.Workload) { w.Spec.Priority = ptr.To(int32(2000)) },
			expectPriority:         1000,
			expectValidationErrors: false,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
				features.GangScheduling:  true,
			})

			ctx := genericapirequest.NewDefaultContext()
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

			if new.Spec.Priority != nil && tt.expectPriority != *new.Spec.Priority {
				t.Errorf("Expected priority = %v, got %v", tt.expectPriority, new.Spec.Priority)
			}
		})
	}
}
