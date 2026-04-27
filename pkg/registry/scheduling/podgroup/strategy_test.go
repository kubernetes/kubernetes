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

package podgroup

import (
	"context"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
)

var podGroup = &scheduling.PodGroup{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "foo",
		Namespace: metav1.NamespaceDefault,
	},
	Spec: scheduling.PodGroupSpec{
		PodGroupTemplateRef: &scheduling.PodGroupTemplateReference{
			Workload: &scheduling.WorkloadPodGroupTemplateReference{
				WorkloadName:         "w",
				PodGroupTemplateName: "t",
			},
		},
		SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
			Gang: &scheduling.GangSchedulingPolicy{
				MinCount: 5,
			},
		},
	},
}

func podGroupWithSchedulingConstraints(keys ...string) *scheduling.PodGroup {
	pg := podGroup.DeepCopy()
	pg.Spec.SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
		Topology: []scheduling.TopologyConstraint{},
	}
	for _, key := range keys {
		constraint := scheduling.TopologyConstraint{Key: key}
		pg.Spec.SchedulingConstraints.Topology = append(pg.Spec.SchedulingConstraints.Topology, constraint)
	}
	return pg
}

func podGroupWithDisruptionMode(mode scheduling.DisruptionMode) *scheduling.PodGroup {
	pg := podGroup.DeepCopy()
	pg.Spec.DisruptionMode = &mode
	return pg
}

var (
	fieldImmutableError    = "field is immutable"
	minCountError          = "must be greater than or equal to 1"
	oneOfError             = "must specify one of: `basic`, `gang`"
	multipleFieldsSetError = "must specify exactly one of: `basic`, `gang`"
	tooManyItemsError      = "must have at most 1 item"
	maximumError           = "must be less than or equal to 1000000000"
	subdomainNameError     = "lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"
	forbiddenError         = "Forbidden"
	supportedModesError    = `supported values: "Pod", "PodGroup"`
)

func TestStrategy(t *testing.T) {
	strategy := NewStrategy()
	if !strategy.NamespaceScoped() {
		t.Errorf("PodGroup must be namespace scoped")
	}
	if strategy.AllowCreateOnUpdate() {
		t.Errorf("PodGroup should not allow create on update")
	}
}

func ctxWithRequestInfo() context.Context {
	return genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        "v1alpha2",
		Resource:          "podgroups",
		IsResourceRequest: true,
	})
}

func TestStrategyCreate(t *testing.T) {
	ctx := ctxWithRequestInfo()
	now := metav1.Now()
	testCases := map[string]struct {
		obj                           *scheduling.PodGroup
		expectObj                     *scheduling.PodGroup
		enableTopologyAwareScheduling bool
		enableWorkloadAwarePreemption bool
		expectValidationError         string
	}{
		"simple": {
			obj:       podGroup,
			expectObj: podGroup,
		},
		"negative min count": {
			obj: func() *scheduling.PodGroup {
				newPodGroup := podGroup.DeepCopy()
				newPodGroup.Spec.SchedulingPolicy.Gang.MinCount = -1
				return newPodGroup
			}(),
			expectValidationError: minCountError,
		},
		"two scheduling policies": {
			obj: func() *scheduling.PodGroup {
				newPodGroup := podGroup.DeepCopy()
				newPodGroup.Spec.SchedulingPolicy.Basic = &scheduling.BasicSchedulingPolicy{}
				return newPodGroup
			}(),
			expectValidationError: multipleFieldsSetError,
		},
		"no scheduling policy is set": {
			obj: func() *scheduling.PodGroup {
				newPodGroup := podGroup.DeepCopy()
				newPodGroup.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{}
				return newPodGroup
			}(),
			expectValidationError: oneOfError,
		},
		"drop status on creation": {
			obj: func() *scheduling.PodGroup {
				newPodGroup := podGroup.DeepCopy()
				newPodGroup.Status.Conditions = []metav1.Condition{
					{
						Type:               scheduling.PodGroupScheduled,
						Status:             metav1.ConditionFalse,
						Reason:             scheduling.PodGroupReasonUnschedulable,
						Message:            "Test status condition message",
						LastTransitionTime: now,
					},
				}
				return newPodGroup
			}(),
			expectObj: podGroup,
		},
		"multiple topology constraints, topology aware scheduling enabled": {
			obj:                           podGroupWithSchedulingConstraints("foo", "bar"),
			enableTopologyAwareScheduling: true,
			expectValidationError:         tooManyItemsError,
		},
		"multiple topology constraints, topology aware scheduling disabled": {
			obj:       podGroupWithSchedulingConstraints("foo", "bar"),
			expectObj: podGroup,
		},
		"invalid topology key, topology aware scheduling enabled": {
			obj:                           podGroupWithSchedulingConstraints("foo-"),
			enableTopologyAwareScheduling: true,
			expectValidationError:         "Invalid value: \"foo-\"",
		},
		"invalid topology key, topology aware scheduling disabled": {
			obj:       podGroupWithSchedulingConstraints("foo-"),
			expectObj: podGroup,
		},
		"with TAS feature gate disabled, drop scheduling constraints on creation": {
			obj:       podGroupWithSchedulingConstraints("foo-"),
			expectObj: podGroup,
		},
		"workload aware preemption disabled - drop disruption mode": {
			obj:       podGroupWithDisruptionMode(scheduling.DisruptionModePod),
			expectObj: podGroup,
		},
		"workload aware preemption enabled - preserve disruption mode (pod)": {
			obj:                           podGroupWithDisruptionMode(scheduling.DisruptionModePod),
			expectObj:                     podGroupWithDisruptionMode(scheduling.DisruptionModePod),
			enableWorkloadAwarePreemption: true,
		},
		"workload aware preemption enabled - preserve disruption mode (pod group)": {
			obj:                           podGroupWithDisruptionMode(scheduling.DisruptionModePodGroup),
			expectObj:                     podGroupWithDisruptionMode(scheduling.DisruptionModePodGroup),
			enableWorkloadAwarePreemption: true,
		},
		"workload aware preemption enabled - unknown disruption mode": {
			obj:                           podGroupWithDisruptionMode(scheduling.DisruptionMode("Invalid")),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         supportedModesError,
		},
		"workload aware preemption disabled - drop priorityClassName": {
			obj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.PriorityClassName = "high-priority"
				return pg
			}(),
			expectObj: podGroup,
		},
		"workload aware preemption enabled - invalid priorityClassName": {
			obj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.PriorityClassName = "invalid/priority/class/name"
				return pg
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         subdomainNameError,
		},
		"workload aware preemption enabled - preserve priorityClassName": {
			obj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.PriorityClassName = "high-priority"
				return pg
			}(),
			expectObj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.PriorityClassName = "high-priority"
				return pg
			}(),
			enableWorkloadAwarePreemption: true,
		},
		"workload aware preemption disabled - drop priority": {
			obj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.Priority = new(int32(1000))
				return pg
			}(),
			expectObj: podGroup,
		},
		"workload aware preemption enabled - preserve priority": {
			obj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.Priority = new(int32(1000))
				return pg
			}(),
			expectObj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.Priority = new(int32(1000))
				return pg
			}(),
			enableWorkloadAwarePreemption: true,
		},
		"workload aware preemption enabled - too high priority": {
			obj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.Priority = new(int32(scheduling.HighestUserDefinablePriority + 1))
				return pg
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         maximumError,
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
			podGroup := tc.obj.DeepCopy()

			strategy := NewStrategy()
			strategy.PrepareForCreate(ctx, podGroup)
			if errs := strategy.Validate(ctx, podGroup); len(errs) != 0 {
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
			if warnings := strategy.WarningsOnCreate(ctx, podGroup); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			strategy.Canonicalize(podGroup)
			if tc.expectObj != nil {
				if diff := cmp.Diff(tc.expectObj, podGroup); diff != "" {
					t.Errorf("got unexpected podGroup object (-want, +got): %s", diff)
				}
			}
		})
	}
}

func TestStrategyUpdate(t *testing.T) {
	ctx := ctxWithRequestInfo()
	testCases := map[string]struct {
		oldObj                        *scheduling.PodGroup
		newObj                        *scheduling.PodGroup
		enableTopologyAwareScheduling bool
		enableWorkloadAwarePreemption bool
		expectValidationError         string
	}{
		"no changes": {
			oldObj: podGroup,
			newObj: podGroup,
		},
		"name change not allowed": {
			oldObj: podGroup,
			newObj: func() *scheduling.PodGroup {
				newPodGroup := podGroup.DeepCopy()
				newPodGroup.Name += "bar"
				return newPodGroup
			}(),
			expectValidationError: fieldImmutableError,
		},
		"updating pod group template ref not allowed": {
			oldObj: podGroup,
			newObj: func() *scheduling.PodGroup {
				newPodGroup := podGroup.DeepCopy()
				newPodGroup.Spec.PodGroupTemplateRef = &scheduling.PodGroupTemplateReference{
					Workload: &scheduling.WorkloadPodGroupTemplateReference{
						WorkloadName:         "foo",
						PodGroupTemplateName: "baz",
					},
				}
				return newPodGroup
			}(),
			expectValidationError: fieldImmutableError,
		},
		"changing min count in gang scheduling policy not allowed": {
			oldObj: podGroup,
			newObj: func() *scheduling.PodGroup {
				newPodGroup := podGroup.DeepCopy()
				newPodGroup.Spec.SchedulingPolicy.Gang.MinCount = 4
				return newPodGroup
			}(),
			expectValidationError: fieldImmutableError,
		},
		"changing scheduling policy not allowed": {
			oldObj: podGroup,
			newObj: func() *scheduling.PodGroup {
				newPodGroup := podGroup.DeepCopy()
				newPodGroup.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
					Basic: &scheduling.BasicSchedulingPolicy{},
				}
				return newPodGroup
			}(),
			expectValidationError: fieldImmutableError,
		},
		"changing scheduling constraints not allowed": {
			oldObj:                        podGroupWithSchedulingConstraints("foo"),
			newObj:                        podGroupWithSchedulingConstraints(),
			enableTopologyAwareScheduling: true,
			expectValidationError:         fieldImmutableError,
		},
		"changing topology constraints not allowed": {
			oldObj:                        podGroupWithSchedulingConstraints("foo"),
			newObj:                        podGroupWithSchedulingConstraints(),
			enableTopologyAwareScheduling: true,
			expectValidationError:         fieldImmutableError,
		},
		"changing topology key not allowed": {
			oldObj:                        podGroupWithSchedulingConstraints("foo"),
			newObj:                        podGroupWithSchedulingConstraints("foobar"),
			enableTopologyAwareScheduling: true,
			expectValidationError:         fieldImmutableError,
		},
		"changing scheduling constraints not allowed with TAS disabled": {
			oldObj:                podGroupWithSchedulingConstraints("foo"),
			newObj:                podGroupWithSchedulingConstraints(),
			expectValidationError: forbiddenError,
		},
		"changing topology constraints not allowed with TAS disabled": {
			oldObj:                podGroupWithSchedulingConstraints("foo"),
			newObj:                podGroupWithSchedulingConstraints(),
			expectValidationError: forbiddenError,
		},
		"changing topology key not allowed with TAS disabled": {
			oldObj:                podGroupWithSchedulingConstraints("foo"),
			newObj:                podGroupWithSchedulingConstraints("foobar"),
			expectValidationError: forbiddenError,
		},
		"disruption mode update, workload aware preemption disabled": {
			oldObj:                podGroupWithDisruptionMode(scheduling.DisruptionModePod),
			newObj:                podGroupWithDisruptionMode(scheduling.DisruptionModePodGroup),
			expectValidationError: forbiddenError,
		},
		"disruption mode update, workload aware preemption enabled": {
			oldObj:                        podGroupWithDisruptionMode(scheduling.DisruptionModePod),
			newObj:                        podGroupWithDisruptionMode(scheduling.DisruptionModePodGroup),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         fieldImmutableError,
		},
		"priority class name update, workload aware preemption disabled": {
			oldObj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.PriorityClassName = "low-priority"
				return pg
			}(),
			newObj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.PriorityClassName = "high-priority"
				return pg
			}(),
			expectValidationError: forbiddenError,
		},
		"priority class name update, workload aware preemption enabled": {
			oldObj: podGroupWithDisruptionMode(scheduling.DisruptionModePod),
			newObj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.PriorityClassName = "high-priority"
				return pg
			}(),
			enableWorkloadAwarePreemption: true,
			expectValidationError:         fieldImmutableError,
		},
		"priority update, workload aware preemption disabled": {
			oldObj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.Priority = new(int32(1000))
				return pg
			}(),
			newObj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.Priority = new(int32(2000))
				return pg
			}(),
			expectValidationError: forbiddenError,
		},
		"priority update, workload aware preemption enabled": {
			oldObj: podGroupWithDisruptionMode(scheduling.DisruptionModePod),
			newObj: func() *scheduling.PodGroup {
				pg := podGroupWithDisruptionMode(scheduling.DisruptionModePod)
				pg.Spec.Priority = new(int32(2000))
				return pg
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
			podGroup := tc.oldObj.DeepCopy()
			newPodGroup := tc.newObj.DeepCopy()
			newPodGroup.ResourceVersion = "4"

			strategy := NewStrategy()
			strategy.PrepareForUpdate(ctx, newPodGroup, podGroup)
			if errs := strategy.ValidateUpdate(ctx, newPodGroup, podGroup); len(errs) != 0 {
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
		})
	}
}

func TestStatusStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:    "scheduling.k8s.io",
		APIVersion:  "v1alpha2",
		Resource:    "podgroups",
		Subresource: "status",
	})
	now := metav1.Now()
	testCases := map[string]struct {
		oldObj                *scheduling.PodGroup
		newObj                *scheduling.PodGroup
		expectObj             *scheduling.PodGroup
		expectValidationError string
	}{
		"no changes": {
			oldObj:    podGroup,
			newObj:    podGroup,
			expectObj: podGroup,
		},
		"valid status change": {
			oldObj: podGroup,
			newObj: func() *scheduling.PodGroup {
				podGroup := podGroup.DeepCopy()
				podGroup.Status.Conditions = append(podGroup.Status.Conditions, metav1.Condition{
					Type:               scheduling.PodGroupScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             scheduling.PodGroupReasonUnschedulable,
					Message:            "Test status condition message",
					LastTransitionTime: now,
				})
				return podGroup
			}(),
			expectObj: func() *scheduling.PodGroup {
				podGroup := podGroup.DeepCopy()
				podGroup.Status.Conditions = append(podGroup.Status.Conditions, metav1.Condition{
					Type:               scheduling.PodGroupScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             scheduling.PodGroupReasonUnschedulable,
					Message:            "Test status condition message",
					LastTransitionTime: now,
				})
				return podGroup
			}(),
		},
		"name change not allowed": {
			oldObj: podGroup,
			newObj: func() *scheduling.PodGroup {
				podGroup := podGroup.DeepCopy()
				podGroup.Name += "-2"
				return podGroup
			}(),
			expectValidationError: fieldImmutableError,
		},
		// Cannot add finalizers, annotations and labels during status update.
		"drop meta changes": {
			oldObj: podGroup,
			newObj: func() *scheduling.PodGroup {
				podGroup := podGroup.DeepCopy()
				podGroup.Finalizers = []string{"foo"}
				podGroup.Annotations = map[string]string{"foo": "bar"}
				podGroup.Labels = map[string]string{"foo": "bar"}
				return podGroup
			}(),
			expectObj: podGroup,
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			strategy := NewStrategy()

			statusStrategy := NewStatusStrategy(strategy)

			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			statusStrategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := statusStrategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
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
			if warnings := statusStrategy.WarningsOnUpdate(ctx, newObj, oldObj); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			statusStrategy.Canonicalize(newObj)

			expectObj := tc.expectObj.DeepCopy()
			expectObj.ResourceVersion = "4"
			if diff := cmp.Diff(expectObj, newObj); diff != "" {
				t.Errorf("PodGroup mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDropPodGroupTemplateResourceClaims(t *testing.T) {
	var noPodGroup *scheduling.PodGroup
	podGroupWithoutClaims := podGroup

	t.Run("spec", func(t *testing.T) {
		podGroupWithClaims := func() *scheduling.PodGroup {
			w := podGroupWithoutClaims.DeepCopy()
			w.Spec.ResourceClaims = []scheduling.PodGroupResourceClaim{
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
			oldPodGroup  *scheduling.PodGroup
			newPodGroup  *scheduling.PodGroup
			wantPodGroup *scheduling.PodGroup
		}{
			{
				description:  "old with claims / new with claims / disabled",
				oldPodGroup:  podGroupWithClaims,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithClaims,
			},
			{
				description:  "old without claims / new with claims / disabled",
				oldPodGroup:  podGroupWithoutClaims,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "no old PodGroup / new with claims / disabled",
				oldPodGroup:  noPodGroup,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithoutClaims,
			},

			{
				description:  "old with claims / new without claims / disabled",
				oldPodGroup:  podGroupWithClaims,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "old without claims / new without claims / disabled",
				oldPodGroup:  podGroupWithoutClaims,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "no old PodGroup / new without claims / disabled",
				oldPodGroup:  noPodGroup,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},

			{
				description:  "old with claims / new with claims / enabled",
				enabled:      true,
				oldPodGroup:  podGroupWithClaims,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithClaims,
			},
			{
				description:  "old without claims / new with claims / enabled",
				enabled:      true,
				oldPodGroup:  podGroupWithoutClaims,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithClaims,
			},
			{
				description:  "no old PodGroup / new with claims / enabled",
				enabled:      true,
				oldPodGroup:  noPodGroup,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithClaims,
			},

			{
				description:  "old with claims / new without claims / enabled",
				enabled:      true,
				oldPodGroup:  podGroupWithClaims,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "old without claims / new without claims / enabled",
				enabled:      true,
				oldPodGroup:  podGroupWithoutClaims,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "no old PodGroup / new without claims / enabled",
				enabled:      true,
				oldPodGroup:  noPodGroup,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
		}

		for _, tc := range tests {
			t.Run(tc.description, func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.DRAWorkloadResourceClaims: tc.enabled,
					features.GenericWorkload:           tc.enabled,
				})

				oldPodGroup := tc.oldPodGroup.DeepCopy()
				newPodGroup := tc.newPodGroup.DeepCopy()
				wantPodGroup := tc.wantPodGroup
				dropDisabledPodGroupFields(newPodGroup, oldPodGroup)

				// old PodGroup should never be changed
				if diff := cmp.Diff(oldPodGroup, tc.oldPodGroup); diff != "" {
					t.Errorf("old PodGroup changed: %s", diff)
				}

				if diff := cmp.Diff(wantPodGroup, newPodGroup); diff != "" {
					t.Errorf("new PodGroup changed (- want, + got): %s", diff)
				}
			})
		}
	})

	t.Run("status", func(t *testing.T) {
		podGroupWithClaims := func() *scheduling.PodGroup {
			w := podGroupWithoutClaims.DeepCopy()
			w.Spec.ResourceClaims = []scheduling.PodGroupResourceClaim{
				{
					Name:              "my-claim",
					ResourceClaimName: new("resource-claim"),
				},
			}
			w.Status.ResourceClaimStatuses = []scheduling.PodGroupResourceClaimStatus{
				{
					Name:              "my-claim",
					ResourceClaimName: new("generated-claim"),
				},
			}
			return w
		}()

		tests := []struct {
			description  string
			enabled      bool
			oldPodGroup  *scheduling.PodGroup
			newPodGroup  *scheduling.PodGroup
			wantPodGroup *scheduling.PodGroup
		}{
			{
				description:  "old with claims / new with claims / disabled",
				oldPodGroup:  podGroupWithClaims,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithClaims,
			},
			{
				description:  "old without claims / new with claims / disabled",
				oldPodGroup:  podGroupWithoutClaims,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "no old PodGroup / new with claims / disabled",
				oldPodGroup:  noPodGroup,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithoutClaims,
			},

			{
				description:  "old with claims / new without claims / disabled",
				oldPodGroup:  podGroupWithClaims,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "old without claims / new without claims / disabled",
				oldPodGroup:  podGroupWithoutClaims,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "no old PodGroup / new without claims / disabled",
				oldPodGroup:  noPodGroup,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},

			{
				description:  "old with claims / new with claims / enabled",
				enabled:      true,
				oldPodGroup:  podGroupWithClaims,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithClaims,
			},
			{
				description:  "old without claims / new with claims / enabled",
				enabled:      true,
				oldPodGroup:  podGroupWithoutClaims,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithClaims,
			},
			{
				description:  "no old PodGroup / new with claims / enabled",
				enabled:      true,
				oldPodGroup:  noPodGroup,
				newPodGroup:  podGroupWithClaims,
				wantPodGroup: podGroupWithClaims,
			},

			{
				description:  "old with claims / new without claims / enabled",
				enabled:      true,
				oldPodGroup:  podGroupWithClaims,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "old without claims / new without claims / enabled",
				enabled:      true,
				oldPodGroup:  podGroupWithoutClaims,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
			{
				description:  "no old PodGroup / new without claims / enabled",
				enabled:      true,
				oldPodGroup:  noPodGroup,
				newPodGroup:  podGroupWithoutClaims,
				wantPodGroup: podGroupWithoutClaims,
			},
		}

		for _, tc := range tests {
			t.Run(tc.description, func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.DRAWorkloadResourceClaims: tc.enabled,
					features.GenericWorkload:           tc.enabled,
				})

				oldPodGroup := tc.oldPodGroup.DeepCopy()
				newPodGroup := tc.newPodGroup.DeepCopy()
				wantPodGroup := tc.wantPodGroup
				dropDisabledPodGroupStatusFields(newPodGroup, oldPodGroup)

				// old PodGroup should never be changed
				if diff := cmp.Diff(oldPodGroup, tc.oldPodGroup); diff != "" {
					t.Errorf("old PodGroup changed: %s", diff)
				}

				if diff := cmp.Diff(wantPodGroup.Status, newPodGroup.Status); diff != "" {
					t.Errorf("new PodGroup changed (- want, + got): %s", diff)
				}
			})
		}
	})
}
