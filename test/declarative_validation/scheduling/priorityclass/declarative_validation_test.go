/*
<<<<<<< HEAD
Copyright The Kubernetes Authors.
=======
Copyright 2025 The Kubernetes Authors.
>>>>>>> cebac4f8d32 (feat(scheduling): migrate PriorityClass.Value to declarative validation)

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

package priorityclass

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
<<<<<<< HEAD
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	scheduling "k8s.io/kubernetes/pkg/apis/scheduling"
	registry "k8s.io/kubernetes/pkg/registry/scheduling/priorityclass"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

// TODO: remove this apiVersions variable once coverage tests are generated for this package.
var apiVersions = []string{"v1", "v1beta1"}
=======
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	registry "k8s.io/kubernetes/pkg/registry/scheduling/priorityclass"

	// Ensure all API groups are registered with the scheme.
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)
>>>>>>> cebac4f8d32 (feat(scheduling): migrate PriorityClass.Value to declarative validation)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

<<<<<<< HEAD
=======
func setValue(v int32) func(obj *scheduling.PriorityClass) {
	return func(obj *scheduling.PriorityClass) {
		obj.Value = v
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "priorityclasses",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidPriorityClass(),
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}
}

>>>>>>> cebac4f8d32 (feat(scheduling): migrate PriorityClass.Value to declarative validation)
func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

<<<<<<< HEAD
func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "priorityclasses",
		IsResourceRequest: true,
		Verb:              "create",
	})

	obj := mkPriorityClass()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "priorityclasses",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})

	updateObj := mkPriorityClass()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkPriorityClass(tweaks ...func(pc *scheduling.PriorityClass)) scheduling.PriorityClass {
	pc := scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-obj",
		},
=======
func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       scheduling.PriorityClass
		updateObj    scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidPriorityClass(),
			updateObj: mkValidPriorityClass(),
		},
		"value changed": {
			oldObj:    mkValidPriorityClass(),
			updateObj: mkValidPriorityClass(setValue(20)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"value set to unset": {
			oldObj:    mkValidPriorityClass(),
			updateObj: mkValidPriorityClass(setValue(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"value unset to set": {
			oldObj:    mkValidPriorityClass(setValue(0)),
			updateObj: mkValidPriorityClass(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "scheduling.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "priorityclasses",
				Name:              "valid-priority-class",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}
}

func mkValidPriorityClass(tweaks ...func(*scheduling.PriorityClass)) scheduling.PriorityClass {
	pc := scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-priority-class",
		},
		Value: 10,
>>>>>>> cebac4f8d32 (feat(scheduling): migrate PriorityClass.Value to declarative validation)
	}
	for _, tweak := range tweaks {
		tweak(&pc)
	}
	return pc
}
