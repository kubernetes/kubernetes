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
	apiVersions := []string{"v1alpha1"} // Workload is currently only in v1alpha1
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
		"empty podGroups": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.PodGroups = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroups"), "must have at least one item"),
			},
		},
		"too many podGroups": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.PodGroups = make([]scheduling.PodGroup, scheduling.WorkloadMaxPodGroups+1)
				for i := range obj.Spec.PodGroups {
					obj.Spec.PodGroups[i] = scheduling.PodGroup{
						Name: fmt.Sprintf("group-%d", i),
						Policy: scheduling.PodGroupPolicy{
							Gang: &scheduling.GangSchedulingPolicy{
								MinCount: 1,
							},
						},
					}
				}
			}),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "podGroups"), scheduling.WorkloadMaxPodGroups+1, scheduling.WorkloadMaxPodGroups).WithOrigin("maxItems"),
			},
		},
		"empty podGroup name": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.PodGroups[0].Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(0).Child("name"), "", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')").WithOrigin("format=k8s-short-name"),
			},
		},
		"invalid podGroup name": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.PodGroups[0].Name = "Invalid_Name"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(0).Child("name"), "Invalid_Name", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')").WithOrigin("format=k8s-short-name"),
			},
		},
		"duplicate podGroup names": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.PodGroups = append(obj.Spec.PodGroups, scheduling.PodGroup{
					Name: "main",
					Policy: scheduling.PodGroupPolicy{
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 1,
						},
					},
				})
			}),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "podGroups").Index(1).Child("name"), "main"),
			},
		},
		"gang minCount zero": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.PodGroups[0].Policy.Gang.MinCount = 0
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(0).Child("policy", "gang", "minCount"), int64(0), "").WithOrigin("minimum"),
			},
		},
		"gang minCount negative": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.PodGroups[0].Policy.Gang.MinCount = -1
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(0).Child("policy", "gang", "minCount"), int64(-1), "").WithOrigin("minimum"),
			},
		},
		"valid with controllerRef": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "apps",
					Kind:     "Deployment",
					Name:     "my-deployment",
				}
			}),
		},
		"controllerRef with empty APIGroup": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "",
					Kind:     "Pod",
					Name:     "my-pod",
				}
			}),
		},
		"controllerRef invalid APIGroup": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "invalid_api_group",
					Kind:     "Deployment",
					Name:     "my-deployment",
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "controllerRef", "apiGroup"), "invalid_api_group", "").WithOrigin("format=k8s-long-name"),
			},
		},
		"controllerRef missing kind": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "apps",
					Kind:     "",
					Name:     "my-deployment",
				}
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "controllerRef", "kind"), ""),
			},
		},
		"controllerRef missing name": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "apps",
					Kind:     "Deployment",
					Name:     "",
				}
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "controllerRef", "name"), ""),
			},
		},
		"controllerRef invalid name": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "apps",
					Kind:     "Deployment",
					Name:     "/invalid-name",
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "controllerRef", "name"), "/invalid-name", "must not contain '/'").WithOrigin("format=k8s-short-name"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha1"} // Workload is currently only in v1alpha1
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
			oldObj:    mkValidWorkload(func(obj *scheduling.Workload) { obj.ResourceVersion = "1" }),
			updateObj: mkValidWorkload(func(obj *scheduling.Workload) { obj.ResourceVersion = "1" }),
		},
		"valid update with unchanged controllerRef": {
			oldObj: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.ResourceVersion = "1"
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "apps",
					Kind:     "Deployment",
					Name:     "my-deployment",
				}
			}),
			updateObj: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.ResourceVersion = "1"
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "apps",
					Kind:     "Deployment",
					Name:     "my-deployment",
				}
			}),
		},
		"invalid update empty podGroups": {
			oldObj: mkValidWorkload(func(obj *scheduling.Workload) { obj.ResourceVersion = "1" }),
			updateObj: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.ResourceVersion = "1"
				obj.Spec.PodGroups = []scheduling.PodGroup{}
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroups"), "must have at least one item"),
				field.Invalid(field.NewPath("spec", "podGroups"), []scheduling.PodGroup{}, "field is immutable"),
			},
		},
		"invalid update too many podGroups": {
			oldObj: mkValidWorkload(func(obj *scheduling.Workload) { obj.ResourceVersion = "1" }),
			updateObj: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.ResourceVersion = "1"
				obj.Spec.PodGroups = make([]scheduling.PodGroup, scheduling.WorkloadMaxPodGroups+1)
				for i := range obj.Spec.PodGroups {
					obj.Spec.PodGroups[i] = scheduling.PodGroup{
						Name: fmt.Sprintf("group-%d", i),
						Policy: scheduling.PodGroupPolicy{
							Gang: &scheduling.GangSchedulingPolicy{
								MinCount: 1,
							},
						},
					}
				}
			}),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "podGroups"), scheduling.WorkloadMaxPodGroups+1, scheduling.WorkloadMaxPodGroups).WithOrigin("maxItems"),
				field.Invalid(field.NewPath("spec", "podGroups"), nil, "field is immutable"),
			},
		},
		"invalid update podGroups": {
			oldObj: mkValidWorkload(func(obj *scheduling.Workload) { obj.ResourceVersion = "1" }),
			updateObj: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.ResourceVersion = "1"
				obj.Spec.PodGroups = append(obj.Spec.PodGroups, scheduling.PodGroup{
					Name: "worker1",
					Policy: scheduling.PodGroupPolicy{
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 1,
						},
					},
				})
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups"), nil, "field is immutable"),
			},
		},
		"invalid update controllerRef": {
			oldObj: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.ResourceVersion = "1"
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "apps",
					Kind:     "Deployment",
					Name:     "my-deployment",
				}
			}),
			updateObj: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.ResourceVersion = "1"
				obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
					APIGroup: "apps",
					Kind:     "Deployment",
					Name:     "different-deployment",
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "controllerRef"), nil, "field is immutable"),
			},
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
			PodGroups: []scheduling.PodGroup{
				{
					Name: "main",
					Policy: scheduling.PodGroupPolicy{
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
