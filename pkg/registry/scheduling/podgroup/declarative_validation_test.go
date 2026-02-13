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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"

	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

func TestDeclarativeValidate(t *testing.T) {
	apiVersions := []string{"v1alpha2"} // PodGroup is currently only in v1alpha2
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "podgroups",
		IsResourceRequest: true,
		Verb:              "create",
	})
	strategy := NewStrategy()

	testCases := map[string]struct {
		input        scheduling.PodGroup
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidPodGroup(),
		},
		"valid with basic policy": {
			input: mkValidPodGroup(setBasicPolicy()),
		},
		// Declarative validation treats 0 as "missing" and returns Required error
		// instead of checking minimum constraint and returning Invalid error.
		"gang minCount zero": {
			input:        mkValidPodGroup(setPodGroupMinCount(0)),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "schedulingPolicy", "gang", "minCount"), "").MarkAlpha()},
		},
		"gang minCount negative": {
			input:        mkValidPodGroup(setPodGroupMinCount(-1)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy", "gang", "minCount"), -1, "").WithOrigin("minimum").MarkAlpha()},
		},
		"no podGroupTemplateRef": {
			input:        mkValidPodGroup(unsetPodGroupTemplateRef()),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplateRef"), "").MarkAlpha()},
		},
		"empty podGroupTemplateRef": {
			input:        mkValidPodGroup(setEmptyPodGroupTemplateRef()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), "", "must specify one of: `workload`").WithOrigin("union").MarkAlpha()},
		},
		"podGroupTemplateRef with empty template name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef("", "workload")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplateRef", "workload", "podGroupTemplateName"), "").MarkAlpha()},
		},
		"podGroupTemplateRef invalid template name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef("temp/late", "workload")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "workload", "podGroupTemplateName"), nil, "").WithOrigin("format=k8s-short-name").MarkAlpha()},
		},
		"podGroupTemplateRef with empty workload name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef("template", "")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplateRef", "workload", "workloadName"), "").MarkAlpha()},
		},
		"podGroupTemplateRef invalid workload name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef("template", "work/load")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "workload", "workloadName"), nil, "").WithOrigin("format=k8s-short-name").MarkAlpha()},
		},
		"policy with neither basic nor gang": {
			input:        mkValidPodGroup(clearPodGroupPolicy()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("union").MarkAlpha()},
		},
		"policy with both basic and gang": {
			input:        mkValidPodGroup(setBothPolicies()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("union").MarkAlpha()},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha2"} // PodGroup is currently only in v1alpha2
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "podgroups",
		Name:              "valid-podgroup",
		IsResourceRequest: true,
		Verb:              "update",
	})
	testCases := map[string]struct {
		oldObj       scheduling.PodGroup
		updateObj    scheduling.PodGroup
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1")),
		},
		"invalid update empty podGroupTemplateRef": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), setEmptyPodGroupTemplateRef()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update unset podGroupTemplateRef": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), unsetPodGroupTemplateRef()),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroupTemplateRef"), "").MarkAlpha(),
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update setPodGroupTemplateRef": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), setPodGroupTemplateRef("other-template", "other-workload")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update with neither basic nor gang": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), clearPodGroupPolicy()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update with both basic and gang": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), setBothPolicies()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update from gang to basic policy": {
			oldObj:       mkValidPodGroup(setResourceVersion("1")),
			updateObj:    mkValidPodGroup(setResourceVersion("1"), setBasicPolicy()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha()},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			strategy := NewStrategy()
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha2"} // PodGroup is currently only in v1alpha2
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateStatusUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateStatusUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:   "apis",
		APIGroup:    "scheduling.k8s.io",
		APIVersion:  apiVersion,
		Resource:    "podgroups",
		Subresource: "status",
		Verb:        "update",
	})
	testCases := map[string]struct {
		oldObj       scheduling.PodGroup
		updateObj    scheduling.PodGroup
		expectedErrs field.ErrorList
	}{
		"valid noop update": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1")),
		},
		"valid status update": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), addCondition(scheduling.PodGroupConditionTypeScheduled)),
		},
		"duplicate condition types": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), addCondition(scheduling.PodGroupConditionTypeScheduled), addCondition(scheduling.PodGroupConditionTypeScheduled)),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("status", "conditions").Index(1).Child("type"), scheduling.PodGroupConditionTypeScheduled).MarkFromImperative(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			strategy := NewStatusStrategy(NewStrategy())
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// mkValidPodGroup produces a PodGroup which passes validation with no tweaks.
func mkValidPodGroup(tweaks ...func(pg *scheduling.PodGroup)) scheduling.PodGroup {
	obj := scheduling.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "podgroup", Namespace: "ns"},
		Spec: scheduling.PodGroupSpec{
			PodGroupTemplateRef: &scheduling.PodGroupTemplateReference{
				Workload: &scheduling.WorkloadPodGroupTemplateReference{
					WorkloadName:         "workload",
					PodGroupTemplateName: "template",
				},
			},
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{
					MinCount: 5,
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func setResourceVersion(v string) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.ResourceVersion = v
	}
}

func unsetPodGroupTemplateRef() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.PodGroupTemplateRef = nil
	}
}

func setEmptyPodGroupTemplateRef() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.PodGroupTemplateRef = &scheduling.PodGroupTemplateReference{}
	}
}

func setPodGroupMinCount(min int) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.SchedulingPolicy.Gang.MinCount = int32(min)
	}
}

func clearPodGroupPolicy() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{}
	}
}

func setBasicPolicy() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
		}
	}
}

func setBothPolicies() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
			Gang:  &scheduling.GangSchedulingPolicy{MinCount: 1},
		}
	}
}

func setPodGroupTemplateRef(templateName, workloadName string) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.PodGroupTemplateRef = &scheduling.PodGroupTemplateReference{
			Workload: &scheduling.WorkloadPodGroupTemplateReference{
				PodGroupTemplateName: templateName,
				WorkloadName:         workloadName,
			},
		}
	}
}

func addCondition(conditionType string) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Status.Conditions = append(obj.Status.Conditions, metav1.Condition{
			Type:               conditionType,
			Status:             metav1.ConditionTrue,
			Reason:             scheduling.PodGroupConditionScheduled,
			Message:            "Test status condition message",
			LastTransitionTime: metav1.Now(),
		})
	}
}
