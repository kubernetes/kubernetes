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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
)

var (
	workload = &scheduling.Workload{
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

	podDisruptionMode      = new(scheduling.DisruptionModePod)
	podGroupDisruptionMode = new(scheduling.DisruptionModePodGroup)
	invalidDisruptionMode  = new(scheduling.DisruptionMode("Invalid"))
)

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

func TestWorkloadStrategyCreate(t *testing.T) {
	ctx := ctxWithRequestInfo()

	testCases := []struct {
		description                   string
		workload                      *scheduling.Workload
		enableWorkloadAwarePreemption bool
		expectValidationError         bool
		expectWorkload                *scheduling.Workload
	}{
		{
			description:    "simple",
			workload:       workload,
			expectWorkload: workload,
		},
		{
			description: "failed validation",
			workload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = -1
				return w
			}(),
			expectValidationError: true,
		},
		{
			description: "workload aware preemption disabled - drop disruption mode",
			workload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = podDisruptionMode
				return w
			}(),
			expectWorkload: workload,
		},
		{
			description: "workload aware preemption enabled - preserve disruption mode (pod)",
			workload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = podDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = podDisruptionMode
				return w
			}(),
		},
		{
			description: "workload aware preemption enabled - preserve disruption mode (pod group)",
			workload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = podGroupDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = podGroupDisruptionMode
				return w
			}(),
		},
		{
			description: "workload aware preemption enabled - unknown disruption mode",
			workload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = invalidDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			workload := tc.workload.DeepCopy()

			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:         true,
				features.GangScheduling:          tc.enableWorkloadAwarePreemption,
				features.WorkloadAwarePreemption: tc.enableWorkloadAwarePreemption,
			})

			Strategy.PrepareForCreate(ctx, workload)
			if errs := Strategy.Validate(ctx, workload); len(errs) != 0 {
				if !tc.expectValidationError {
					t.Fatalf("Unexpected validation error: %v", errs)
				}
				return
			}
			if tc.expectValidationError {
				t.Fatal("expected validation error(s), got none")
			}
			if warnings := Strategy.WarningsOnCreate(ctx, workload); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			Strategy.Canonicalize(workload)
			if diff := cmp.Diff(tc.expectWorkload, workload); diff != "" {
				t.Errorf("Workload mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestWorkloadStrategyUpdate(t *testing.T) {
	ctx := ctxWithRequestInfo()

	testCases := []struct {
		description                   string
		oldWorkload                   *scheduling.Workload
		newWorkload                   *scheduling.Workload
		enableWorkloadAwarePreemption bool
		expectValidationError         bool
		expectWorkload                *scheduling.Workload
	}{
		{
			description:    "no changes",
			oldWorkload:    workload,
			newWorkload:    workload,
			expectWorkload: workload,
		},
		{
			description: "name update",
			oldWorkload: workload,
			newWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Name += "bar"
				return w
			}(),
			expectValidationError: true,
		},
		{
			description: "invalid spec update - controllerRef",
			oldWorkload: workload,
			newWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					Kind: "foo",
					Name: "baz",
				}
				return w
			}(),
			expectValidationError: true,
		},
		{
			description: "invalid spec update - podGroupTemplates",
			oldWorkload: workload,
			newWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = 4
				return w
			}(),
			expectValidationError: true,
		},
		{
			description: "disruption mode update, workload aware preemption disabled",
			oldWorkload: workload,
			newWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = podDisruptionMode
				return w
			}(),
			expectWorkload:        workload,
			expectValidationError: true,
		},
		{
			description: "disruption mode update, workload aware preemption enabled",
			oldWorkload: workload,
			newWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = podDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			oldWorkload := tc.oldWorkload.DeepCopy()
			newWorkload := tc.newWorkload.DeepCopy()
			newWorkload.ResourceVersion = "4"

			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:         true,
				features.GangScheduling:          tc.enableWorkloadAwarePreemption,
				features.WorkloadAwarePreemption: tc.enableWorkloadAwarePreemption,
			})

			Strategy.PrepareForUpdate(ctx, newWorkload, oldWorkload)
			if errs := Strategy.ValidateUpdate(ctx, newWorkload, oldWorkload); errs != nil {
				if !tc.expectValidationError {
					t.Fatalf("unexpected validation error: %q", errs)
				}
				return
			}
			if tc.expectValidationError {
				t.Fatal("expected validation error(s), got none")
			}
			if warnings := Strategy.WarningsOnUpdate(ctx, newWorkload, oldWorkload); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			Strategy.Canonicalize(newWorkload)

			expectWorkload := tc.expectWorkload.DeepCopy()
			expectWorkload.ResourceVersion = "4"
			if diff := cmp.Diff(expectWorkload, newWorkload); diff != "" {
				t.Errorf("Workload mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
