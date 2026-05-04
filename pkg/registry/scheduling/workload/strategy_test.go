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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/api/operation"
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

	podDisruptionMode      = scheduling.DisruptionModePod
	podGroupDisruptionMode = scheduling.DisruptionModePodGroup
	invalidDisruptionMode  = scheduling.DisruptionMode("Invalid")

	fieldImmutableError = "field is immutable"
	minCountError       = "must be greater than or equal to 1"
	tooManyItemsError   = "must have at most 1 item"
	requiredError       = "Required value"
	subdomainNameError  = "lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"
	supportedModesError = `supported values: "Pod", "PodGroup"`
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

func TestStrategyCreate(t *testing.T) {
	ctx := ctxWithRequestInfo()

	testCases := map[string]struct {
		obj                           *scheduling.Workload
		expectObj                     *scheduling.Workload
		enableTopologyAwareScheduling bool
		enableWorkloadAwarePreemption bool
		expectValidationError         string
	}{
		"simple": {
			obj:       workload,
			expectObj: workload,
		},
		"failed validation": {
			obj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = -1
				return w
			}(),
			expectValidationError: minCountError,
		},
		"drops field with SchedulingConstraints set and TAS disabled": {
			obj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			expectObj: workload,
		},
		"valid with SchedulingConstraints set and TAS enabled": {
			obj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			expectObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			enableTopologyAwareScheduling: true,
		},
		"invalid with multiple topology constraints": {
			obj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{
						{Key: "foo"},
						{Key: "bar"},
					},
				}
				return workload
			}(),
			enableTopologyAwareScheduling: true,
			expectValidationError:         tooManyItemsError,
		},
		"invalid with invalid topology key": {
			obj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{
						{Key: ""},
					},
				}
				return workload
			}(),
			enableTopologyAwareScheduling: true,
			expectValidationError:         requiredError,
		},
		"workload aware preemption disabled - drop disruption mode": {
			obj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = &podDisruptionMode
				return w
			}(),
			expectObj: workload,
		},
		"workload aware preemption enabled - preserve disruption mode (pod)": {
			obj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = &podDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = &podDisruptionMode
				return w
			}(),
		},
		"workload aware preemption enabled - preserve disruption mode (pod group)": {
			obj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = &podGroupDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = &podGroupDisruptionMode
				return w
			}(),
		},
		"workload aware preemption enabled - unknown disruption mode": {
			obj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = &invalidDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         supportedModesError,
		},
		"workload aware preemption enabled - preserve priorityClassName": {
			obj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].PriorityClassName = "high-priority"
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].PriorityClassName = "high-priority"
				return w
			}(),
		},
		"workload aware preemption disabled - drop priorityClassName": {
			obj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].PriorityClassName = "high-priority"
				return w
			}(),
			expectObj: workload,
		},
		"workload aware preemption enabled - invalid priorityClassName": {
			obj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].PriorityClassName = "invalid/priority/class/name"
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         subdomainNameError,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			workload := tc.obj.DeepCopy()

			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: tc.enableTopologyAwareScheduling,
				features.GangScheduling:                  tc.enableWorkloadAwarePreemption,
				features.WorkloadAwarePreemption:         tc.enableWorkloadAwarePreemption,
			})

			Strategy.PrepareForCreate(ctx, workload)
			errs := Strategy.Validate(ctx, workload)
			errs = Strategy.ValidateDeclaratively(ctx, workload, nil, errs, operation.Create, Strategy.DeclarativeValidationConfig(ctx, workload, nil))
			if len(errs) != 0 {
				if tc.expectValidationError == "" {
					t.Fatalf("unexpected error(s): %v", errs)
				}
				if len(errs) != 1 {
					t.Fatalf("exactly one error expected")
				}
				if errMsg := errs[0].Error(); !strings.Contains(errMsg, tc.expectValidationError) {
					t.Fatalf("error %#v does not contain the expected message %q", errMsg, tc.expectValidationError)
				}
			}
			if tc.expectObj != nil {
				if diff := cmp.Diff(tc.expectObj, workload); diff != "" {
					t.Errorf("got unexpected workload object (-want, +got): %s", diff)
				}
			}
		})
	}
}

func TestStrategyUpdate(t *testing.T) {
	ctx := ctxWithRequestInfo()

	testCases := map[string]struct {
		oldObj                        *scheduling.Workload
		newObj                        *scheduling.Workload
		enableTopologyAwareScheduling bool
		enableWorkloadAwarePreemption bool
		expectValidationError         string
		expectWorkload                *scheduling.Workload
	}{
		"no changes": {
			oldObj:         workload,
			newObj:         workload,
			expectWorkload: workload,
		},
		"name update": {
			oldObj: workload,
			newObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Name += "bar"
				return w
			}(),
			expectValidationError: fieldImmutableError,
		},
		"invalid spec update - controllerRef": {
			oldObj: workload,
			newObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					Kind: "foo",
					Name: "baz",
				}
				return w
			}(),
			expectValidationError: fieldImmutableError,
		},
		"invalid spec update - podGroupTemplates": {
			oldObj: workload,
			newObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = 4
				return w
			}(),
			expectValidationError: fieldImmutableError,
		},
		"valid update with scheduling constraints unchanged and TAS disabled": {
			oldObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			newObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			expectWorkload: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
		},
		"valid update with scheduling constraints unchanged and TAS enabled": {
			oldObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			newObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			expectWorkload: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			enableTopologyAwareScheduling: true,
		},
		"changing topology key not allowed with TAS disabled": {
			oldObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			newObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "bar"}},
				}
				return workload
			}(),
			expectValidationError: fieldImmutableError,
		},
		"changing topology key not allowed with TAS enabled": {
			oldObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			newObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "bar"}},
				}
				return workload
			}(),
			enableTopologyAwareScheduling: true,
			expectValidationError:         fieldImmutableError,
		},
		"topology constraint addition is dropped with TAS disabled": {
			oldObj: workload,
			newObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			expectWorkload: workload,
		},
		"changing topology constraints not allowed with TAS enabled": {
			oldObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			newObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				workload.Spec.PodGroupTemplates[1].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			enableTopologyAwareScheduling: true,
			expectValidationError:         fieldImmutableError,
		},
		"adding scheduling constraints not allowed with TAS disabled": {
			oldObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			newObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				workload.Spec.PodGroupTemplates[1].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			expectValidationError: fieldImmutableError,
		},
		"adding scheduling constraints not allowed with TAS enabled": {
			oldObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			newObj: func() *scheduling.Workload {
				workload := workload.DeepCopy()
				workload.Spec.PodGroupTemplates = append(workload.Spec.PodGroupTemplates, *workload.Spec.PodGroupTemplates[0].DeepCopy())
				workload.Spec.PodGroupTemplates[0].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				workload.Spec.PodGroupTemplates[1].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{{Key: "foo"}},
				}
				return workload
			}(),
			enableTopologyAwareScheduling: true,
			expectValidationError:         fieldImmutableError,
		},
		"disruption mode update, workload aware preemption disabled": {
			oldObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = &podGroupDisruptionMode
				return w
			}(),
			newObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = &podDisruptionMode
				return w
			}(),
			expectValidationError: fieldImmutableError,
		},
		"disruption mode update, workload aware preemption enabled": {
			oldObj: workload,
			newObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].DisruptionMode = &podDisruptionMode
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         fieldImmutableError,
		},
		"priorityClassName update, workload aware preemption disabled": {
			oldObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].PriorityClassName = "high-priority"
				return w
			}(),
			newObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].PriorityClassName = "low-priority"
				return w
			}(),
			expectValidationError: fieldImmutableError,
		},
		"priorityClassName update, workload aware preemption enabled": {
			oldObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].PriorityClassName = "high-priority"
				return w
			}(),
			newObj: func() *scheduling.Workload {
				w := workload.DeepCopy()
				w.Spec.PodGroupTemplates[0].PriorityClassName = "low-priority"
				return w
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         fieldImmutableError,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: tc.enableTopologyAwareScheduling,
				features.GangScheduling:                  tc.enableWorkloadAwarePreemption,
				features.WorkloadAwarePreemption:         tc.enableWorkloadAwarePreemption,
			})
			oldWorkload := tc.oldObj.DeepCopy()
			newWorkload := tc.newObj.DeepCopy()
			newWorkload.ResourceVersion = "4"

			Strategy.PrepareForUpdate(ctx, newWorkload, oldWorkload)
			errs := Strategy.ValidateUpdate(ctx, newWorkload, oldWorkload)
			errs = Strategy.ValidateDeclaratively(ctx, newWorkload, oldWorkload, errs, operation.Update, Strategy.DeclarativeValidationConfig(ctx, newWorkload, oldWorkload))
			if len(errs) != 0 {
				if tc.expectValidationError == "" {
					t.Fatalf("unexpected error(s): %v", errs)
				}
				if len(errs) != 1 {
					t.Fatalf("exactly one error expected")
				}
				if errMsg := errs[0].Error(); !strings.Contains(errMsg, tc.expectValidationError) {
					t.Fatalf("error %#v does not contain the expected message %q", errMsg, tc.expectValidationError)
				}
				return
			}
			if tc.expectValidationError != "" {
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
			dropDisabledWorkloadFields(newWorkload, oldWorkload)

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
