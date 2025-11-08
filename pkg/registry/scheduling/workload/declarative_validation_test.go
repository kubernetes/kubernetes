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
				field.Invalid(field.NewPath("spec", "podGroups").Index(0).Child("name"), "", ""),
			},
		},
		"invalid podGroup name": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				obj.Spec.PodGroups[0].Name = "Invalid_Name"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(0).Child("name"), "Invalid_Name", ""),
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
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
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
