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
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
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
		PriorityClassName: ptr.To("low-priority"),
		Priority:          ptr.To(int32(1000)),
	},
}

var (
	fieldImmutableError    = "field is immutable"
	minCountError          = "must be greater than or equal to 1"
	oneOfError             = "must specify one of: `basic`, `gang`"
	multipleFieldsSetError = "must specify exactly one of: `basic`, `gang`"
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
		obj                   *scheduling.PodGroup
		expectObj             *scheduling.PodGroup
		expectValidationError string
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
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
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
		})
	}
}

func TestStrategyUpdate(t *testing.T) {
	ctx := ctxWithRequestInfo()
	testCases := map[string]struct {
		oldObj                *scheduling.PodGroup
		newObj                *scheduling.PodGroup
		expectValidationError string
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
		"changing priorityClassName not allowed": {
			oldObj: podGroup,
			newObj: func() *scheduling.PodGroup {
				pg := podGroup.DeepCopy()
				pg.Spec.PriorityClassName = ptr.To("high-priority")
				return pg
			}(),
			expectValidationError: fieldImmutableError,
		},
		"changing priority not allowed": {
			oldObj: podGroup,
			newObj: func() *scheduling.PodGroup {
				pg := podGroup.DeepCopy()
				pg.Spec.Priority = ptr.To(int32(2000))
				return pg
			}(),
			expectValidationError: fieldImmutableError,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
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

func TestPodGroupStrategyCreate(t *testing.T) {
	testCases := []struct {
		name                          string
		preparePodGroup               func(*scheduling.PodGroup)
		enableWorkloadAwarePreemption bool
		expectPriorityClassName       *string
		expectPriority                *int32
	}{
		{
			name: "priorityClassName is preserved upon creating a pod group when workload-aware-preemption is enabled",
			preparePodGroup: func(pg *scheduling.PodGroup) {
				pg.Spec.PriorityClassName = ptr.To("high-priority")
			},
			enableWorkloadAwarePreemption: true,
			expectPriorityClassName:       ptr.To("high-priority"),
			expectPriority:                ptr.To(int32(1000)),
		},
		{
			name: "priorityClassName and priority are cleared upon creating a pod group when workload-aware-preemption is disabled",
			preparePodGroup: func(pg *scheduling.PodGroup) {
				pg.Spec.PriorityClassName = ptr.To("high-priority")
				pg.Spec.Priority = ptr.To(int32(1000))
			},
			enableWorkloadAwarePreemption: false,
			expectPriorityClassName:       nil,
			expectPriority:                nil,
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
			pg := podGroup.DeepCopy()
			tt.preparePodGroup(pg)

			strategy := NewStrategy()
			strategy.PrepareForCreate(ctx, pg)

			if !reflect.DeepEqual(tt.expectPriorityClassName, pg.Spec.PriorityClassName) {
				t.Errorf("Expected priorityClassName = %v, got %v", tt.expectPriorityClassName, pg.Spec.PriorityClassName)
			}
			if !reflect.DeepEqual(tt.expectPriority, pg.Spec.Priority) {
				t.Errorf("Expected priority = %v, got %v", tt.expectPriority, pg.Spec.Priority)
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
			assert.Equal(t, expectObj, newObj)
		})
	}
}
