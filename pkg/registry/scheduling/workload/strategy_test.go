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
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	ptr "k8s.io/utils/ptr"
)

var (
	workload = &scheduling.Workload{
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

	podDisruptionMode      = ptr.To(scheduling.DisruptionModePod)
	podGroupDisruptionMode = ptr.To(scheduling.DisruptionModePodGroup)
	invalidDisruptionMode  = ptr.To(scheduling.DisruptionMode("Invalid"))
)

func TestWorkloadStrategy(t *testing.T) {
	if !Strategy.NamespaceScoped() {
		t.Errorf("Workload must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Workload should not allow create on update")
	}
}

func TestWorkloadStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

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
				w.Spec.PodGroups[0].Policy.Gang.MinCount = -1
				return w
			}(),
			expectValidationError: true,
		},
		{
			description: "workload aware preemption disabled - drop disruption mode",
			workload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroups[0].Policy.Gang.DisruptionMode = podDisruptionMode
				return w
			}(),
			expectWorkload: workload,
		},
		{
			description: "workload aware preemption enabled - preserve disruption mode (pod)",
			workload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroups[0].Policy.Gang.DisruptionMode = podDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroups[0].Policy.Gang.DisruptionMode = podDisruptionMode
				return w
			}(),
		},
		{
			description: "workload aware preemption enabled - preserve disruption mode (pod group)",
			workload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroups[0].Policy.Gang.DisruptionMode = podGroupDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroups[0].Policy.Gang.DisruptionMode = podGroupDisruptionMode
				return w
			}(),
		},
		{
			description: "workload aware preemption enabled - unknown disruption mode",
			workload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroups[0].Policy.Gang.DisruptionMode = invalidDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			workload := tc.workload.DeepCopy()

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			if tc.enableWorkloadAwarePreemption {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GangScheduling, true)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadAwarePreemption, true)
			}

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
			assert.Equal(t, tc.expectWorkload, workload)
		})
	}
}

func TestWorkloadStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

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
			description: "spec update",
			oldWorkload: workload,
			newWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					Kind: "foo",
					Name: "baz",
				}
				return w
			}(),
			expectWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					Kind: "foo",
					Name: "baz",
				}
				return w
			}(),
		},
		{
			description: "invalid spec update",
			oldWorkload: workload,
			newWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroups[0].Policy.Gang.MinCount = 4
				return w
			}(),
			expectValidationError: true,
		},
		{
			description: "disruption mode update, workload aware preemption disabled",
			oldWorkload: workload,
			newWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroups[0].Policy.Gang.DisruptionMode = podDisruptionMode
				return w
			}(),
			expectWorkload: workload,
		},
		{
			description: "disruption mode update, workload aware preemption enabled",
			oldWorkload: workload,
			newWorkload: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroups[0].Policy.Gang.DisruptionMode = podDisruptionMode
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

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			if tc.enableWorkloadAwarePreemption {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GangScheduling, true)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadAwarePreemption, true)
			}

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
			assert.Equal(t, expectWorkload, newWorkload)
		})
	}
}
