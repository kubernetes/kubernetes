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

package compositepodgroup

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

	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

var (
	cpg = &scheduling.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: scheduling.CompositePodGroupSpec{
			WorkloadRef: &scheduling.WorkloadReference{
				WorkloadName: "workload1",
				TemplateName: "cpg-template1",
			},
			SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
				Gang: &scheduling.CompositeGangSchedulingPolicy{
					MinGroupCount: 5,
				},
			},
		},
	}

	fieldImmutableError    = "field is immutable"
	minCountError          = "must be greater than or equal to 1"
	subdomainNameError     = "lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"
	maximumError           = "must be less than or equal to"
	oneOfError             = "must specify one of: `basic`, `gang`"
	multipleFieldsSetError = "must specify exactly one of: `basic`, `gang`"
)

func TestStrategy(t *testing.T) {
	strategy := NewStrategy()
	if !strategy.NamespaceScoped() {
		t.Errorf("CompositePodGroup must be namespace scoped")
	}
	if strategy.AllowCreateOnUpdate(context.Background()) {
		t.Errorf("CompositePodGroup should not allow create on update")
	}
}

func ctxWithRequestInfo() context.Context {
	return genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        "v1alpha3",
		Resource:          "compositepodgroups",
		IsResourceRequest: true,
	})
}

func TestStrategyCreate(t *testing.T) {
	ctx := ctxWithRequestInfo()

	testCases := map[string]struct {
		obj                   *scheduling.CompositePodGroup
		expectObj             *scheduling.CompositePodGroup
		expectValidationError string
	}{
		"simple": {
			obj:       cpg,
			expectObj: cpg,
		},
		"simple with basic scheduling policy": {
			obj: func() *scheduling.CompositePodGroup {
				c := cpg.DeepCopy()
				c.Spec.SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{
					Basic: &scheduling.CompositeBasicSchedulingPolicy{},
				}
				return c
			}(),
			expectObj: func() *scheduling.CompositePodGroup {
				c := cpg.DeepCopy()
				c.Spec.SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{
					Basic: &scheduling.CompositeBasicSchedulingPolicy{},
				}
				return c
			}(),
		},
		"validation error - both basic and gang specified": {
			obj: func() *scheduling.CompositePodGroup {
				c := cpg.DeepCopy()
				c.Spec.SchedulingPolicy.Basic = &scheduling.CompositeBasicSchedulingPolicy{}
				return c
			}(),
			expectValidationError: multipleFieldsSetError,
		},
		"validation error - neither basic nor gang specified": {
			obj: func() *scheduling.CompositePodGroup {
				c := cpg.DeepCopy()
				c.Spec.SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{}
				return c
			}(),
			expectValidationError: oneOfError,
		},
		"failed validation": {
			obj: func() *scheduling.CompositePodGroup {
				newCpg := cpg.DeepCopy()
				newCpg.Spec.SchedulingPolicy.Gang.MinGroupCount = -1
				return newCpg
			}(),
			expectValidationError: minCountError,
		},
		"invalid priorityClassName": {
			obj: func() *scheduling.CompositePodGroup {
				cpg := cpg.DeepCopy()
				cpg.Spec.PriorityClassName = "invalid/priority/class/name"
				return cpg
			}(),
			expectValidationError: subdomainNameError,
		},
		"preserve priorityClassName": {
			obj: func() *scheduling.CompositePodGroup {
				cpg := cpg.DeepCopy()
				cpg.Spec.PriorityClassName = "high-priority"
				return cpg
			}(),
			expectObj: func() *scheduling.CompositePodGroup {
				cpg := cpg.DeepCopy()
				cpg.Spec.PriorityClassName = "high-priority"
				return cpg
			}(),
		},
		"preserve priority": {
			obj: func() *scheduling.CompositePodGroup {
				cpg := cpg.DeepCopy()
				cpg.Spec.Priority = new(int32(1000))
				return cpg
			}(),
			expectObj: func() *scheduling.CompositePodGroup {
				cpg := cpg.DeepCopy()
				cpg.Spec.Priority = new(int32(1000))
				return cpg
			}(),
		},
		"too high priority": {
			obj: func() *scheduling.CompositePodGroup {
				cpg := cpg.DeepCopy()
				cpg.Spec.Priority = new(int32(scheduling.HighestUserDefinablePriority + 1))
				return cpg
			}(),
			expectValidationError: maximumError,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.CompositePodGroup:               true,
				features.TopologyAwareWorkloadScheduling: true,
			})
			newCpg := tc.obj.DeepCopy()

			strategy := NewStrategy()
			strategy.PrepareForCreate(ctx, newCpg)
			errs := strategy.Validate(ctx, newCpg)
			errs = strategy.ValidateDeclaratively(ctx, newCpg, nil, errs, operation.Create, strategy.DeclarativeValidationConfig(ctx, newCpg, nil))
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
			if warnings := strategy.WarningsOnCreate(ctx, newCpg); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			strategy.Canonicalize(newCpg)
			if tc.expectObj != nil {
				if diff := cmp.Diff(tc.expectObj, newCpg); diff != "" {
					t.Errorf("got unexpected cpg object (-want, +got): %s", diff)
				}
			}
		})
	}
}

func TestStrategyUpdate(t *testing.T) {
	ctx := ctxWithRequestInfo()
	testCases := map[string]struct {
		oldObj                *scheduling.CompositePodGroup
		newObj                *scheduling.CompositePodGroup
		expectValidationError string
	}{
		"no changes": {
			oldObj: cpg,
			newObj: cpg,
		},
		"name change not allowed": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				newCpg := cpg.DeepCopy()
				newCpg.Name += "bar"
				return newCpg
			}(),
			expectValidationError: fieldImmutableError,
		},
		"updating workload ref not allowed": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				newCpg := cpg.DeepCopy()
				newCpg.Spec.WorkloadRef = &scheduling.WorkloadReference{
					WorkloadName: "foo",
					TemplateName: "baz",
				}
				return newCpg
			}(),
			expectValidationError: fieldImmutableError,
		},
		"changing parentCompositePodGroupName from nil not allowed": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				newCpg := cpg.DeepCopy()
				parent := "parent-cpg"
				newCpg.Spec.ParentCompositePodGroupName = &parent
				return newCpg
			}(),
			expectValidationError: fieldImmutableError,
		},
		"changing parentCompositePodGroupName from non-nil not allowed": {
			oldObj: func() *scheduling.CompositePodGroup {
				oldCpg := cpg.DeepCopy()
				parent := "parent-cpg"
				oldCpg.Spec.ParentCompositePodGroupName = &parent
				return oldCpg
			}(),
			newObj: func() *scheduling.CompositePodGroup {
				newCpg := cpg.DeepCopy()
				parent := "other-parent-cpg"
				newCpg.Spec.ParentCompositePodGroupName = &parent
				return newCpg
			}(),
			expectValidationError: fieldImmutableError,
		},
		"changing min group count in gang scheduling policy not allowed": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				newCpg := cpg.DeepCopy()
				newCpg.Spec.SchedulingPolicy.Gang.MinGroupCount = 4
				return newCpg
			}(),
			expectValidationError: fieldImmutableError,
		},
		"changing scheduling policy not allowed": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				newCpg := cpg.DeepCopy()
				newCpg.Spec.SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{
					Basic: &scheduling.CompositeBasicSchedulingPolicy{},
				}
				return newCpg
			}(),
			expectValidationError: fieldImmutableError,
		},
		"priority class name update": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				cpg := cpg.DeepCopy()
				cpg.Spec.PriorityClassName = "high-priority"
				return cpg
			}(),
			expectValidationError: fieldImmutableError,
		},
		"priority update": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				cpg := cpg.DeepCopy()
				cpg.Spec.Priority = new(int32(2000))
				return cpg
			}(),
			expectValidationError: fieldImmutableError,
		},
		"parentCompositePodGroupName update not allowed": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				newCpg := cpg.DeepCopy()
				newCpg.Spec.ParentCompositePodGroupName = new("parent1")
				return newCpg
			}(),
			expectValidationError: fieldImmutableError,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.CompositePodGroup:               true,
				features.TopologyAwareWorkloadScheduling: true,
			})
			oldCpg := tc.oldObj.DeepCopy()
			newCpg := tc.newObj.DeepCopy()
			newCpg.ResourceVersion = "4"

			strategy := NewStrategy()
			strategy.PrepareForUpdate(ctx, newCpg, oldCpg)
			errs := strategy.ValidateUpdate(ctx, newCpg, oldCpg)
			errs = strategy.ValidateDeclaratively(ctx, newCpg, oldCpg, errs, operation.Update, strategy.DeclarativeValidationConfig(ctx, newCpg, oldCpg))
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
		})
	}
}

func TestStatusStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:    "scheduling.k8s.io",
		APIVersion:  "v1alpha3",
		Resource:    "compositepodgroups",
		Subresource: "status",
	})
	now := metav1.Now()
	testCases := map[string]struct {
		oldObj                *scheduling.CompositePodGroup
		newObj                *scheduling.CompositePodGroup
		expectObj             *scheduling.CompositePodGroup
		expectValidationError string
	}{
		"no changes": {
			oldObj:    cpg,
			newObj:    cpg,
			expectObj: cpg,
		},
		"valid status change": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				c := cpg.DeepCopy()
				c.Status.Conditions = append(c.Status.Conditions, metav1.Condition{
					Type:               "CompositePodGroupInitiallyScheduled",
					Status:             metav1.ConditionFalse,
					Reason:             "Unschedulable",
					Message:            "Test status condition message",
					LastTransitionTime: now,
				})
				return c
			}(),
			expectObj: func() *scheduling.CompositePodGroup {
				c := cpg.DeepCopy()
				c.Status.Conditions = append(c.Status.Conditions, metav1.Condition{
					Type:               "CompositePodGroupInitiallyScheduled",
					Status:             metav1.ConditionFalse,
					Reason:             "Unschedulable",
					Message:            "Test status condition message",
					LastTransitionTime: now,
				})
				return c
			}(),
		},
		"name change not allowed": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				c := cpg.DeepCopy()
				c.Name += "-2"
				return c
			}(),
			expectValidationError: fieldImmutableError,
		},
		"drop meta and spec changes": {
			oldObj: cpg,
			newObj: func() *scheduling.CompositePodGroup {
				c := cpg.DeepCopy()
				c.Finalizers = []string{"foo"}
				c.Annotations = map[string]string{"foo": "bar"}
				c.Labels = map[string]string{"foo": "bar"}
				c.Spec.SchedulingPolicy.Gang.MinGroupCount = 10
				return c
			}(),
			expectObj: cpg,
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
				t.Errorf("CompositePodGroup mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
