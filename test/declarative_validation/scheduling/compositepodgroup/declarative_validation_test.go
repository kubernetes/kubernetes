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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/test/coverage"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/scheduling/compositepodgroup"
	"k8s.io/kubernetes/test/declarative_validation/meta"

	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

func TestDeclarativeValidate(t *testing.T) {
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
		Resource:          "compositepodgroups",
		IsResourceRequest: true,
		Verb:              "create",
	})
	strategy := registry.NewStrategy()

	testCases := map[string]struct {
		input        scheduling.CompositePodGroup
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidCompositePodGroup(),
		},
		"parentCompositePodGroupName invalid": {
			input:        mkValidCompositePodGroup(setParentCompositePodGroupName("invalid/name")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "parentCompositePodGroupName"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"priority too high": {
			input:        mkValidCompositePodGroup(setPriority(scheduling.HighestUserDefinablePriority + 1)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "priority"), nil, "").WithOrigin("maximum")},
		},
		"priorityClassName invalid": {
			input:        mkValidCompositePodGroup(setPriorityClassName("invalid/name")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "priorityClassName"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"schedulingPolicy invalid union": {
			input:        mkValidCompositePodGroup(clearSchedulingPolicy()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("union")},
		},
		"schedulingPolicy gang minGroupCount zero": {
			input:        mkValidCompositePodGroup(setGangPolicy(0)),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "schedulingPolicy", "gang", "minGroupCount"), "")},
		},
		"schedulingPolicy gang minGroupCount negative": {
			input:        mkValidCompositePodGroup(setGangPolicy(-1)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy", "gang", "minGroupCount"), nil, "").WithOrigin("minimum")},
		},
		"no workloadRef": {
			input:        mkValidCompositePodGroup(clearWorkloadRef()),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "workloadRef"), "")},
		},
		"workloadRef empty templateName": {
			input:        mkValidCompositePodGroup(setWorkloadRef("", "workload")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "workloadRef", "templateName"), "")},
		},
		"workloadRef empty workloadName": {
			input:        mkValidCompositePodGroup(setWorkloadRef("template", "")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "workloadRef", "workloadName"), "")},
		},
		"workloadRef invalid templateName": {
			input:        mkValidCompositePodGroup(setWorkloadRef("invalid/template", "workload")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "workloadRef", "templateName"), nil, "").WithOrigin("format=k8s-short-name")},
		},
		"workloadRef invalid workloadName": {
			input:        mkValidCompositePodGroup(setWorkloadRef("template", "invalid/workload")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "workloadRef", "workloadName"), nil, "").WithOrigin("format=k8s-long-name")},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.CompositePodGroup:               true,
				features.TopologyAwareWorkloadScheduling: true,
			})
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, strategy, tc.expectedErrs)
		})
	}

	obj := mkValidCompositePodGroup()
	meta.RunObjectMetaTestCases(t, ctx, &obj, strategy, meta.WithStringentFinalizerValidation())
}

func TestDeclarativeValidateUpdate(t *testing.T) {
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
		Resource:          "compositepodgroups",
		Name:              "valid-compositepodgroup",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		oldObj       scheduling.CompositePodGroup
		updateObj    scheduling.CompositePodGroup
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1")),
		},
		"invalid parentCompositePodGroupName update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1"), setParentCompositePodGroupName("parent1")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1"), setParentCompositePodGroupName("parent2")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "parentCompositePodGroupName"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid priority update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1"), setPriority(100)),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1"), setPriority(200)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "priority"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid priorityClassName update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1"), setPriorityClassName("low-priority")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1"), setPriorityClassName("high-priority")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "priorityClassName"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid schedulingPolicy update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1"), setGangPolicy(5)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid workloadRef update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1"), clearWorkloadRef()),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "workloadRef"), ""),
				field.Invalid(field.NewPath("spec", "workloadRef"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid workloadRef.templateName update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1"), setWorkloadRef("other-template", "workload")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "workloadRef"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid workloadRef.workloadName update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1"), setWorkloadRef("template", "other-workload")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "workloadRef"), nil, "").WithOrigin("immutable"),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.CompositePodGroup:               true,
				features.TopologyAwareWorkloadScheduling: true,
			})
			strategy := registry.NewStrategy()
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, strategy, tc.expectedErrs)
		})
	}

	updateObj := mkValidCompositePodGroup(setResourceVersion("1"))
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.NewStrategy(), meta.WithStringentFinalizerValidation())

	// Disable checks that are currently impossible to test properly in isolation.
	// FieldValueInvalid update rules for properties under spec.schedulingPolicy
	// are impossible to hit because mutations inside the schedulingPolicy fail its top-level
	// immutable check oldSelf == self, short-circuiting deeper validation.
	gvk := schema.GroupVersionKind{Group: "scheduling.k8s.io", Version: apiVersion, Kind: "CompositePodGroup"}
	coverage.RecordObservedRules(gvk, field.ErrorList{
		field.Invalid(field.NewPath("spec", "schedulingPolicy", "basic"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("spec", "schedulingPolicy", "gang"), nil, "").WithOrigin("update"),
	})
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
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
		Resource:    "compositepodgroups",
		Subresource: "status",
		Verb:        "update",
	})
	testCases := map[string]struct {
		oldObj       scheduling.CompositePodGroup
		updateObj    scheduling.CompositePodGroup
		expectedErrs field.ErrorList
	}{
		"invalid status condition observedGeneration minimum": {
			oldObj: mkValidCompositePodGroup(setResourceVersion("1")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1"), func(obj *scheduling.CompositePodGroup) {
				obj.Status.Conditions = []metav1.Condition{{
					Type:               "Scheduled",
					Status:             metav1.ConditionTrue,
					LastTransitionTime: metav1.Now(),
					Reason:             "Scheduled",
					ObservedGeneration: -1,
				}}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("observedGeneration"), nil, "").WithOrigin("minimum").MarkAlpha(),
			},
		},
		"valid noop update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1")),
		},
		"valid status update": {
			oldObj:    mkValidCompositePodGroup(setResourceVersion("1")),
			updateObj: mkValidCompositePodGroup(setResourceVersion("1"), addCondition("CompositePodGroupInitiallyScheduled")),
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			strategy := registry.NewStatusStrategy(registry.NewStrategy())
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, strategy, tc.expectedErrs, apitesting.WithSubResources("status"))
		})
	}

	meta.RunConditionTestCases(t, ctx, field.NewPath("status", "conditions"), &scheduling.CompositePodGroup{}, registry.NewStatusStrategy(registry.NewStrategy()), func(obj *scheduling.CompositePodGroup, c []metav1.Condition) {
		*obj = mkValidCompositePodGroup(setResourceVersion("1"), func(cpg *scheduling.CompositePodGroup) { cpg.Status.Conditions = c })
	})
}

func mkValidCompositePodGroup(tweaks ...func(cpg *scheduling.CompositePodGroup)) scheduling.CompositePodGroup {
	obj := scheduling.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "cpg", Namespace: "ns"},
		Spec: scheduling.CompositePodGroupSpec{
			WorkloadRef: &scheduling.WorkloadReference{
				WorkloadName: "workload",
				TemplateName: "template",
			},
			SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
				Basic: &scheduling.CompositeBasicSchedulingPolicy{},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func setResourceVersion(v string) func(obj *scheduling.CompositePodGroup) {
	return func(obj *scheduling.CompositePodGroup) {
		obj.ResourceVersion = v
	}
}

func setParentCompositePodGroupName(name string) func(obj *scheduling.CompositePodGroup) {
	return func(obj *scheduling.CompositePodGroup) {
		obj.Spec.ParentCompositePodGroupName = &name
	}
}

func setPriority(priority int32) func(obj *scheduling.CompositePodGroup) {
	return func(obj *scheduling.CompositePodGroup) {
		obj.Spec.Priority = &priority
	}
}

func setPriorityClassName(priorityClassName string) func(obj *scheduling.CompositePodGroup) {
	return func(obj *scheduling.CompositePodGroup) {
		obj.Spec.PriorityClassName = priorityClassName
	}
}

func clearSchedulingPolicy() func(obj *scheduling.CompositePodGroup) {
	return func(obj *scheduling.CompositePodGroup) {
		obj.Spec.SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{}
	}
}

func setGangPolicy(minGroupCount int32) func(obj *scheduling.CompositePodGroup) {
	return func(obj *scheduling.CompositePodGroup) {
		obj.Spec.SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{
			Gang: &scheduling.CompositeGangSchedulingPolicy{
				MinGroupCount: minGroupCount,
			},
		}
	}
}

func clearWorkloadRef() func(obj *scheduling.CompositePodGroup) {
	return func(obj *scheduling.CompositePodGroup) {
		obj.Spec.WorkloadRef = nil
	}
}

func setWorkloadRef(templateName, workloadName string) func(obj *scheduling.CompositePodGroup) {
	return func(obj *scheduling.CompositePodGroup) {
		obj.Spec.WorkloadRef = &scheduling.WorkloadReference{
			TemplateName: templateName,
			WorkloadName: workloadName,
		}
	}
}

func addCondition(conditionType string) func(*scheduling.CompositePodGroup) {
	return func(cpg *scheduling.CompositePodGroup) {
		cpg.Status.Conditions = append(cpg.Status.Conditions, metav1.Condition{
			Type:               conditionType,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
			Reason:             "Reason",
			Message:            "Message",
		})
	}
}
