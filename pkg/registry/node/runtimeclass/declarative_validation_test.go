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
)

func TestDeclarativeValidate(t *testing.T) {
	// RuntimeClass is served as node.k8s.io/v1.
	apiVersions := []string{"v1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIGroup:          "node.k8s.io",
			APIVersion:        apiVersion,
			Resource:          "runtimeclasses",
			IsResourceRequest: true,
			Verb:              "create",
		},
	)

	testCases := map[string]struct {
		input        node.RuntimeClass
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidRuntimeClass(),
		},
		"invalid handler": {
			input: mkValidRuntimeClass(func(obj *node.RuntimeClass) {
				obj.Handler = ""
			}),
			expectedErrs: field.ErrorList{
				// The handwritten validator returns "Invalid value" for "",
				// not "Required value".
				field.Invalid(field.NewPath("handler"), "", ""),
			},
		},
		// TODO: add more cases (e.g. invalid scheduling, invalid overhead)
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(
				t,
				ctx,
				&tc.input,
				Strategy.Validate,
				tc.expectedErrs,
			)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       node.RuntimeClass
		updateObj    node.RuntimeClass
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj: mkValidRuntimeClass(func(obj *node.RuntimeClass) {
				obj.ResourceVersion = "1"
			}),
			updateObj: mkValidRuntimeClass(func(obj *node.RuntimeClass) {
				obj.ResourceVersion = "1"
			}),
		},
		"invalid update handler": {
			oldObj: mkValidRuntimeClass(func(obj *node.RuntimeClass) {
				obj.ResourceVersion = "1"
			}),
			updateObj: mkValidRuntimeClass(func(obj *node.RuntimeClass) {
				obj.ResourceVersion = "1"
				obj.Handler = ""
			}),
			expectedErrs: field.ErrorList{
				// New value "" still has to be a valid RFC1123 label,
				// so we expect "Invalid value".
				field.Invalid(field.NewPath("handler"), "", ""),
				// And handler is immutable, so changing it is forbidden.
				field.Forbidden(field.NewPath("handler"), "updates to handler are forbidden."),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIPrefix:         "apis",
					APIGroup:          "node.k8s.io",
					APIVersion:        apiVersion,
					Resource:          "runtimeclasses",
					Name:              "valid-runtime-class",
					IsResourceRequest: true,
					Verb:              "update",
				},
			)

			apitesting.VerifyUpdateValidationEquivalence(
				t,
				ctx,
				&tc.updateObj,
				&tc.oldObj,
				Strategy.ValidateUpdate,
				tc.expectedErrs,
			)
		})
	}
}

// mkValidRuntimeClass returns a semantically valid RuntimeClass and then applies any tweaks.
func mkValidRuntimeClass(tweaks ...func(obj *node.RuntimeClass)) node.RuntimeClass {
	obj := node.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-runtime-class",
		},
		Handler: "runc",
		// Overhead and Scheduling intentionally omitted for the base "valid" object.
	}

	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
