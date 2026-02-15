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
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

var podGroup = &scheduling.PodGroup{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "foo",
		Namespace: metav1.NamespaceDefault,
	},
	Spec: scheduling.PodGroupSpec{
		SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
			Gang: &scheduling.GangSchedulingPolicy{
				MinCount: 5,
			},
		},
	},
}

var (
	fieldImmutableError    = "field is immutable"
	minCountError          = "must be greater than zero"
	oneOfError             = "must specify one of: `basic`, `gang`"
	multipleFieldsSetError = "exactly one of `basic`, `gang` is required"
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

func TestStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
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
						Type:   "PodGroupScheduled",
						Status: metav1.ConditionTrue,
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
				assert.Len(t, errs, 1, "exactly one error expected")
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
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
	ctx := genericapirequest.NewDefaultContext()
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
					WorkloadName:         "foo",
					PodGroupTemplateName: "baz",
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
				assert.Len(t, errs, 1, "exactly one error expected")
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
				return
			}
			if tc.expectValidationError != "" {
				t.Fatal("expected validation error(s), got none")
			}
		})
	}
}

func TestStatusStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
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
					Type:   "PodGroupScheduled",
					Status: metav1.ConditionTrue,
				})
				return podGroup
			}(),
			expectObj: func() *scheduling.PodGroup {
				podGroup := podGroup.DeepCopy()
				podGroup.Status.Conditions = append(podGroup.Status.Conditions, metav1.Condition{
					Type:   "PodGroupScheduled",
					Status: metav1.ConditionTrue,
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
				assert.Len(t, errs, 1, "exactly one error expected")
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
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
