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
	"fmt"
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
	apiVersions := []string{"v1alpha2"} // Workload is currently only in v1alpha2
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
		Resource:          "workloads",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        scheduling.Workload
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidWorkload(),
		},
		"empty podGroupTemplates": {
			input:        mkValidWorkload(clearPodGroupTemplates()),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates"), "must have at least one item").MarkAlpha()},
		},
		"too many podGroupTemplates": {
			input:        mkValidWorkload(setManyPodGroupTemplates(scheduling.WorkloadMaxPodGroupTemplates + 1)),
			expectedErrs: field.ErrorList{field.TooMany(field.NewPath("spec", "podGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems").MarkAlpha()},
		},
		"empty podGroupTemplate name": {
			input:        mkValidWorkload(setPodGroupName(0, "")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("name"), "").MarkAlpha()},
		},
		"invalid podGroupTemplate name": {
			input:        mkValidWorkload(setPodGroupName(0, "Invalid_Name")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name").MarkAlpha()},
		},
		"duplicate podGroupTemplate names": {
			input:        mkValidWorkload(addPodGroupTemplate("main")),
			expectedErrs: field.ErrorList{field.Duplicate(field.NewPath("spec", "podGroupTemplates").Index(1), nil).MarkAlpha()},
		},
		// Declarative validation treats 0 as "missing" and returns Required error
		// instead of checking minimum constraint and returning Invalid error.
		"gang minCount zero": {
			input:        mkValidWorkload(setPodGroupMinCount(0, 0)),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minCount"), "").MarkAlpha()},
		},
		"gang minCount negative": {
			input:        mkValidWorkload(setPodGroupMinCount(0, -1)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minCount"), nil, "").WithOrigin("minimum").MarkAlpha()},
		},
		"valid with controllerRef": {
			input: mkValidWorkload(setControllerRef("apps", "Deployment", "my-deployment")),
		},
		"controllerRef with empty APIGroup": {
			input: mkValidWorkload(setControllerRef("", "Pod", "my-pod")),
		},
		"controllerRef invalid APIGroup": {
			input:        mkValidWorkload(setControllerRef("invalid_api_group", "Deployment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "apiGroup"), nil, "").WithOrigin("format=k8s-long-name").MarkAlpha()},
		},
		"controllerRef missing kind": {
			input:        mkValidWorkload(setControllerRef("apps", "", "my-deployment")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "controllerRef", "kind"), "").MarkAlpha()},
		},
		"controllerRef invalid kind with slash": {
			input:        mkValidWorkload(setControllerRef("apps", "Deploy/ment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "kind"), nil, "").WithOrigin("format=k8s-path-segment-name").MarkAlpha()},
		},
		"controllerRef missing name": {
			input:        mkValidWorkload(setControllerRef("apps", "Deployment", "")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "controllerRef", "name"), "").MarkAlpha()},
		},
		"controllerRef invalid name": {
			input:        mkValidWorkload(setControllerRef("apps", "Deployment", "/invalid-name")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "name"), nil, "").WithOrigin("format=k8s-path-segment-name").MarkAlpha()},
		},
		"controllerRef invalid kind with percent": {
			input:        mkValidWorkload(setControllerRef("apps", "Deploy%ment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "kind"), nil, "").WithOrigin("format=k8s-path-segment-name").MarkAlpha()},
		},
		"controllerRef invalid name with percent": {
			input:        mkValidWorkload(setControllerRef("apps", "Deployment", "my%deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "name"), nil, "").WithOrigin("format=k8s-path-segment-name").MarkAlpha()},
		},
		"policy with neither basic nor gang": {
			input:        mkValidWorkload(clearPodGroupPolicy(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union").MarkAlpha()},
		},
		"policy with both basic and gang": {
			input:        mkValidWorkload(setBothPolicies(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union").MarkAlpha()},
		},
		"valid with basic policy": {
			input: mkValidWorkload(setBasicPolicy(0)),
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha2"} // Workload is currently only in v1alpha2
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       scheduling.Workload
		updateObj    scheduling.Workload
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1")),
		},
		"valid update with unchanged controllerRef": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
		},
		"valid update with setting controllerRef": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
		},
		"invalid update empty podGroupTemplates": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setEmptyPodGroupTemplates()),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroupTemplates"), "must have at least one item").MarkAlpha(),
				field.Invalid(field.NewPath("spec", "podGroupTemplates"), []scheduling.PodGroupTemplate{}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update too many podGroupTemplates": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setManyPodGroupTemplates(scheduling.WorkloadMaxPodGroupTemplates+1)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "podGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems").MarkAlpha(),
				field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update podGroupTemplate": {
			oldObj:       mkValidWorkload(setResourceVersion("1")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), addPodGroupTemplate("worker1")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha()},
		},
		"invalid update controllerRef": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "different-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef"), nil, "field is immutable").WithOrigin("update").MarkAlpha()},
		},
		"invalid update with neither basic nor gang": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), clearPodGroupPolicy(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update with both basic and gang": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setBothPolicies(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"valid update from gang to basic policy": {
			oldObj:       mkValidWorkload(setResourceVersion("1")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setBasicPolicy(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha()},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "scheduling.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "workloads",
				Name:              "valid-workload",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidWorkload(tweaks ...func(obj *scheduling.Workload)) scheduling.Workload {
	obj := scheduling.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-workload",
			Namespace: "default",
		},
		Spec: scheduling.WorkloadSpec{
			PodGroupTemplates: []scheduling.PodGroupTemplate{
				{
					Name: "main",
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 1,
						},
					},
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func setResourceVersion(v string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.ResourceVersion = v
	}
}

func clearPodGroupTemplates() func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = nil
	}
}

func setEmptyPodGroupTemplates() func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = []scheduling.PodGroupTemplate{}
	}
}

func setManyPodGroupTemplates(n int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = make([]scheduling.PodGroupTemplate, n)
		for i := range obj.Spec.PodGroupTemplates {
			obj.Spec.PodGroupTemplates[i] = scheduling.PodGroupTemplate{
				Name: fmt.Sprintf("group-%d", i),
				SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
					Gang: &scheduling.GangSchedulingPolicy{
						MinCount: 1,
					},
				},
			}
		}
	}
}

func addPodGroupTemplate(name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = append(obj.Spec.PodGroupTemplates, scheduling.PodGroupTemplate{
			Name: name,
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{
					MinCount: 1,
				},
			},
		})
	}
}

func setPodGroupName(pgIdx int, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].Name = name
	}
}

func setPodGroupMinCount(pgIdx, min int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy.Gang.MinCount = int32(min)
	}
}

func clearPodGroupPolicy(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{}
	}
}

func setBasicPolicy(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
		}
	}
}

func setBothPolicies(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
			Gang:  &scheduling.GangSchedulingPolicy{MinCount: 1},
		}
	}
}

func setControllerRef(apiGroup, kind, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
			APIGroup: apiGroup,
			Kind:     kind,
			Name:     name,
		}
	}
}
