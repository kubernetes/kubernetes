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

func TestDropPodGroupTemplateResourceClaims(t *testing.T) {
	var noWorkload *scheduling.Workload
	workloadWithoutClaims := workload
	workloadWithClaims := func() *scheduling.Workload {
		w := workloadWithoutClaims.DeepCopy()
		w.Spec.PodGroupTemplates[0].ResourceClaims = []scheduling.PodGroupResourceClaim{
			{
				Name:              "my-claim",
				ResourceClaimName: new("resource-claim"),
			},
		}
		return w
	}()

	tests := []struct {
		description  string
		enabled      bool
		oldWorkload  *scheduling.Workload
		newWorkload  *scheduling.Workload
		wantWorkload *scheduling.Workload
	}{
		{
			description:  "old with claims / new with claims / disabled",
			oldWorkload:  workloadWithClaims,
			newWorkload:  workloadWithClaims,
			wantWorkload: workloadWithClaims,
		},
		{
			description:  "old without claims / new with claims / disabled",
			oldWorkload:  workloadWithoutClaims,
			newWorkload:  workloadWithClaims,
			wantWorkload: workloadWithoutClaims,
		},
		{
			description:  "no old workload / new with claims / disabled",
			oldWorkload:  noWorkload,
			newWorkload:  workloadWithClaims,
			wantWorkload: workloadWithoutClaims,
		},

		{
			description:  "old with claims / new without claims / disabled",
			oldWorkload:  workloadWithClaims,
			newWorkload:  workloadWithoutClaims,
			wantWorkload: workloadWithoutClaims,
		},
		{
			description:  "old without claims / new without claims / disabled",
			oldWorkload:  workloadWithoutClaims,
			newWorkload:  workloadWithoutClaims,
			wantWorkload: workloadWithoutClaims,
		},
		{
			description:  "no old workload / new without claims / disabled",
			oldWorkload:  noWorkload,
			newWorkload:  workloadWithoutClaims,
			wantWorkload: workloadWithoutClaims,
		},

		{
			description:  "old with claims / new with claims / enabled",
			enabled:      true,
			oldWorkload:  workloadWithClaims,
			newWorkload:  workloadWithClaims,
			wantWorkload: workloadWithClaims,
		},
		{
			description:  "old without claims / new with claims / enabled",
			enabled:      true,
			oldWorkload:  workloadWithoutClaims,
			newWorkload:  workloadWithClaims,
			wantWorkload: workloadWithClaims,
		},
		{
			description:  "no old workload / new with claims / enabled",
			enabled:      true,
			oldWorkload:  noWorkload,
			newWorkload:  workloadWithClaims,
			wantWorkload: workloadWithClaims,
		},

		{
			description:  "old with claims / new without claims / enabled",
			enabled:      true,
			oldWorkload:  workloadWithClaims,
			newWorkload:  workloadWithoutClaims,
			wantWorkload: workloadWithoutClaims,
		},
		{
			description:  "old without claims / new without claims / enabled",
			enabled:      true,
			oldWorkload:  workloadWithoutClaims,
			newWorkload:  workloadWithoutClaims,
			wantWorkload: workloadWithoutClaims,
		},
		{
			description:  "no old workload / new without claims / enabled",
			enabled:      true,
			oldWorkload:  noWorkload,
			newWorkload:  workloadWithoutClaims,
			wantWorkload: workloadWithoutClaims,
		},
	}

	for _, tc := range tests {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.DRAWorkloadResourceClaims: tc.enabled,
				features.GenericWorkload:           tc.enabled,
			})

			oldWorkload := tc.oldWorkload.DeepCopy()
			newWorkload := tc.newWorkload.DeepCopy()
			wantWorkload := tc.wantWorkload
			dropDisabledFields(newWorkload, oldWorkload)

			// old Workload should never be changed
			if diff := cmp.Diff(oldWorkload, tc.oldWorkload); diff != "" {
				t.Errorf("old Workload changed: %s", diff)
			}

			if diff := cmp.Diff(wantWorkload, newWorkload); diff != "" {
				t.Errorf("new Workload changed (- want, + got): %s", diff)
			}
		})
	}
}
