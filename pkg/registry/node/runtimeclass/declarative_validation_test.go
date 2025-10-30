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

package runtimeclass

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	node "k8s.io/kubernetes/pkg/apis/node"
)

func mkValidRuntimeClassRequired(tweaks ...func(rc *node.RuntimeClass)) node.RuntimeClass {
	rc := node.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "myrc",
		},

		// handler must still be DNSLabel, lowercase, immutable, etc.
		Handler: "runc",

		// Overhead is now required.
		Overhead: &node.Overhead{
			PodFixed: api.ResourceList{
				api.ResourceCPU: resource.MustParse("100m"),
				// memory can be added if needed, CPU alone is fine for validity
			},
		},

		// Scheduling is now required.
		// Minimal-but-valid Scheduling: give it a nodeSelector.
		Scheduling: &node.Scheduling{
			NodeSelector: map[string]string{
				"kubernetes.io/arch": "amd64",
			},
		},
	}

	for _, tweak := range tweaks {
		tweak(&rc)
	}
	return rc
}

func TestRuntimeClassValidateRequiredForDeclarative(t *testing.T) {
	apiVersions := []string{"v1", "v1beta1"}

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:   "node.k8s.io",
					APIVersion: apiVersion,
				},
			)

			testCases := map[string]struct {
				obj          node.RuntimeClass
				expectedErrs field.ErrorList
			}{
				"all required fields present": {
					obj:          mkValidRuntimeClassRequired(),
					expectedErrs: field.ErrorList{},
				},

				"missing overhead only": {
					obj: mkValidRuntimeClassRequired(func(rc *node.RuntimeClass) {
						rc.Overhead = nil
					}),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("overhead"), ""),
					},
				},

				"missing scheduling only": {
					obj: mkValidRuntimeClassRequired(func(rc *node.RuntimeClass) {
						rc.Scheduling = nil
					}),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("scheduling"), ""),
					},
				},

				"missing both overhead and scheduling": {
					obj: mkValidRuntimeClassRequired(func(rc *node.RuntimeClass) {
						rc.Overhead = nil
						rc.Scheduling = nil
					}),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("overhead"), ""),
						field.Required(field.NewPath("scheduling"), ""),
					},
				},
			}

			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					// CREATE path: Strategy.Validate
					apitesting.VerifyValidationEquivalence(
						t,
						ctx,
						&tc.obj,
						Strategy.Validate,
						tc.expectedErrs,
					)
				})
			}
		})
	}
}

func TestRuntimeClassValidateImmutableHandlerForDeclarative(t *testing.T) {
	apiVersions := []string{"v1", "v1beta1"}

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:   "node.k8s.io",
					APIVersion: apiVersion,
				},
			)

			testCases := map[string]struct {
				oldObj       node.RuntimeClass
				newObj       node.RuntimeClass
				expectedErrs field.ErrorList
			}{
				"no-op update (same handler)": {
					oldObj: mkValidRuntimeClassRequired(func(rc *node.RuntimeClass) {
						rc.Handler = "runc"
					}),
					newObj: mkValidRuntimeClassRequired(func(rc *node.RuntimeClass) {
						rc.Handler = "runc"
					}),
					expectedErrs: field.ErrorList{},
				},

				"handler changed is rejected (immutable)": {
					oldObj: mkValidRuntimeClassRequired(func(rc *node.RuntimeClass) {
						rc.Handler = "runc"
					}),
					newObj: mkValidRuntimeClassRequired(func(rc *node.RuntimeClass) {
						rc.Handler = "gvisor"
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(
							field.NewPath("handler"),
							"gvisor",
							apivalidation.FieldImmutableErrorMsg,
						).MarkCoveredByDeclarative(),
					},
				},
			}

			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					// To make this an UPDATE and not a CREATE, both objects
					// must have the same name and a non-empty ResourceVersion.
					tc.oldObj.ObjectMeta.Name = "myrc"
					tc.newObj.ObjectMeta.Name = "myrc"
					tc.oldObj.ObjectMeta.ResourceVersion = "1"
					tc.newObj.ObjectMeta.ResourceVersion = "1"

					apitesting.VerifyUpdateValidationEquivalence(
						t,
						ctx,
						&tc.newObj,
						&tc.oldObj,
						Strategy.ValidateUpdate,
						tc.expectedErrs,
					)
				})
			}
		})
	}
}
