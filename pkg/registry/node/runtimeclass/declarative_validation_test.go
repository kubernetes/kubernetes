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

	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	node "k8s.io/kubernetes/pkg/apis/node"
)

func mkRuntimeClassHandlerOnly(tweaks ...func(*node.RuntimeClass)) node.RuntimeClass {
	rc := node.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "myrc",
		},
		// valid dnsLabel
		Handler: "runc",
	}
	for _, f := range tweaks {
		f(&rc)
	}
	return rc
}

func TestRuntimeClass_DeclarativeValidate_Handler(t *testing.T) {
	apiVersions := []string{"v1"}

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:   "node.k8s.io",
					APIVersion: apiVersion,
				},
			)

			tests := map[string]struct {
				obj          node.RuntimeClass
				expectedErrs field.ErrorList
			}{
				//"valid": {
				//	obj:          mkRuntimeClassHandlerOnly(),
				//	expectedErrs: field.ErrorList{},
				//},
				//"missing handler": {
				//	obj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
				//		rc.Handler = ""
				//	}),
				//	expectedErrs: field.ErrorList{
				//		field.Required(field.NewPath("handler"), ""),
				//	},
				//},
				"invalid handler (not dnsLabel)": {
					obj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
						rc.Handler = "Not-Valid" // uppercase + hyphen
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("handler"), "Not-Valid", "must be a DNS label"),
					},
				},
			}

			for name, tc := range tests {
				t.Run(name, func(t *testing.T) {
					// this checks: handwritten == declarative
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

func TestRuntimeClass_DeclarativeValidate_ImmutableHandler(t *testing.T) {
	apiVersions := []string{"v1"}

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:   "node.k8s.io",
					APIVersion: apiVersion,
				},
			)

			tests := map[string]struct {
				oldObj, newObj node.RuntimeClass
				expectedErrs   field.ErrorList
			}{
				"no-op update": {
					oldObj:       mkRuntimeClassHandlerOnly(),
					newObj:       mkRuntimeClassHandlerOnly(),
					expectedErrs: field.ErrorList{},
				},
				"handler changed (immutable)": {
					oldObj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
						rc.Handler = "runc"
					}),
					newObj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
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
				//"old empty -> new set (allow for legacy objs)": {
				//	oldObj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
				//		rc.Handler = ""
				//	}),
				//	newObj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
				//		rc.Handler = "runc"
				//	}),
				//	expectedErrs: field.ErrorList{},
				//},
			}

			for name, tc := range tests {
				t.Run(name, func(t *testing.T) {
					// make it UPDATE
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
