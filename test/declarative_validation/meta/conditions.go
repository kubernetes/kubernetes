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

package meta

import (
	"context"
	"testing"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
)

// ConditionTestCase contains a conditions slice and the expected validation errors it should produce.
type ConditionTestCase struct {
	Name         string
	Conditions   []metav1.Condition
	ExpectedErrs field.ErrorList
}

// GenerateConditionTestCases returns a standard set of declarative validation test cases
// for a field of type []metav1.Condition that is annotated with listType=map and listMapKey=type.
// fldPath is the path to the conditions field in the object being validated.
func GenerateConditionTestCases(fldPath *field.Path) []ConditionTestCase {
	return []ConditionTestCase{
		{
			Name: "invalid duplicate types",
			Conditions: []metav1.Condition{
				MkCondition(),
				MkCondition(TweakStatus(metav1.ConditionFalse)),
			},
			ExpectedErrs: field.ErrorList{
				field.Duplicate(fldPath.Index(1), "").MarkAlpha(),
			},
		},
		{
			Name: "invalid missing type",
			Conditions: []metav1.Condition{
				MkCondition(TweakType("")),
			},
			ExpectedErrs: field.ErrorList{
				field.Required(fldPath.Index(0).Child("type"), "").MarkAlpha(),
			},
		},
		{
			Name: "invalid negative observedGeneration",
			Conditions: []metav1.Condition{
				MkCondition(TweakObservedGeneration(-1)),
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(
					fldPath.Index(0).Child("observedGeneration"),
					int64(-1),
					"",
				).WithOrigin("minimum").MarkAlpha(),
			},
		},
		{
			Name: "valid observedGeneration zero",
			Conditions: []metav1.Condition{
				MkCondition(TweakObservedGeneration(0)),
			},
			ExpectedErrs: nil,
		},
	}
}

func RunConditionTestCases[T runtime.Object](t *testing.T, ctx context.Context, fldPath *field.Path, baseObj T, strategy rest.RESTUpdateStrategy, setConditions func(T, []metav1.Condition)) {
	t.Helper()
	for _, tc := range GenerateConditionTestCases(fldPath) {
		t.Run("conditions: "+tc.Name, func(t *testing.T) {
			obj := baseObj.DeepCopyObject().(T)
			setConditions(obj, tc.Conditions)
			if accessor, err := apimeta.Accessor(obj); err == nil {
				accessor.SetResourceVersion("1")
			}
			old := obj.DeepCopyObject().(T)
			setConditions(old, nil)
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, obj, old, strategy, tc.ExpectedErrs, apitesting.WithSubResources("status"))
		})
	}
}

func MkCondition(tweaks ...func(*metav1.Condition)) metav1.Condition {
	c := metav1.Condition{
		Type:               "Ready",
		Status:             metav1.ConditionTrue,
		Reason:             "Foo",
		Message:            "Bar",
		LastTransitionTime: metav1.Unix(1, 0),
	}
	for _, tweak := range tweaks {
		tweak(&c)
	}
	return c
}

func TweakType(condType string) func(*metav1.Condition) {
	return func(c *metav1.Condition) {
		c.Type = condType
	}
}

func TweakStatus(status metav1.ConditionStatus) func(*metav1.Condition) {
	return func(c *metav1.Condition) {
		c.Status = status
	}
}

func TweakObservedGeneration(gen int64) func(*metav1.Condition) {
	return func(c *metav1.Condition) {
		c.ObservedGeneration = gen
	}
}
