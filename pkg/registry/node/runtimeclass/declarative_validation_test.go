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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	node "k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/apis/node/validation"
)

var apiVersions = []string{"v1", "v1beta1", "v1alpha1"}

func mkRuntimeClassHandlerOnly(tweaks ...func(*node.RuntimeClass)) node.RuntimeClass {
	rc := node.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "myrc",
		},
		Handler: "runc",
	}
	for _, f := range tweaks {
		f(&rc)
	}
	return rc
}

func TestRuntimeClass_DeclarativeValidate_Create(t *testing.T) {
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
				"valid handler": {
					obj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
						rc.Handler = "test"
					}),
					expectedErrs: field.ErrorList{},
				},
				"empty handler": {
					obj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
						rc.Handler = ""
					}),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("handler"), "must be a valid DNS label"),
					},
				},
				"handler with special characters": {
					obj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
						rc.Handler = "asasdasda&^%"
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("handler"),
							"", "").WithOrigin("format=k8s-short-name"),
					},
				},
				"handler with uppercase and special characters": {
					obj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
						rc.Handler = "asasdasda&^%&^%$UUUUUUU"
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("handler"),
							"", "").WithOrigin("format=k8s-short-name"),
					},
				},
				"handler exceeds length with invalid characters": {
					obj: mkRuntimeClassHandlerOnly(func(rc *node.RuntimeClass) {
						rc.Handler = "asasdasda&^%&^%$UUUUUUUaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbcccccccccccccccccccccccc"
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("handler"),
							"", "").WithOrigin("format=k8s-short-name"),
					},
				},
			}

			for name, tc := range tests {
				t.Run(name, func(t *testing.T) {
					apitesting.VerifyValidationEquivalence(t, ctx, &tc.obj, Strategy.Validate, tc.expectedErrs,
						apitesting.WithNormalizationRules(validation.NodeNormalizationRules...))
				})
			}
		})
	}
}

func TestRuntimeClass_DeclarativeValidate_Update(t *testing.T) {
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
						rc.Handler = "asadsasdas"
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("handler"), "",
							"").WithOrigin("immutable"),
					},
				},
			}

			for name, tc := range tests {
				t.Run(name, func(t *testing.T) {
					tc.oldObj.ObjectMeta.ResourceVersion = "1"
					tc.newObj.ObjectMeta.ResourceVersion = "1"
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.newObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs, apitesting.WithNormalizationRules(validation.NodeNormalizationRules...))
				})
			}
		})
	}
}
