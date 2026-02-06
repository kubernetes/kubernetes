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

package controllerrevision

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
)

var apiVersions = []string{"v1", "v1beta1", "v1beta2"}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			testDeclarativeValidateForDeclarative(t, v)
		})
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIGroup:   "apps",
			APIVersion: apiVersion,
			Resource:   "controllerrevisions",
		},
	)

	testCases := map[string]struct {
		input        apps.ControllerRevision
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidControllerRevision(),
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			testValidateUpdateForDeclarative(t, v)
		})
	}
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIGroup:   "apps",
			APIVersion: apiVersion,
			Resource:   "controllerrevisions",
		},
	)

	testCases := map[string]struct {
		old, update  apps.ControllerRevision
		expectedErrs field.ErrorList
	}{
		"valid update": {
			old:    mkValidControllerRevision(),
			update: mkValidControllerRevision(),
		},
		"invalid update changing data": {
			old:    mkValidControllerRevision(),
			update: mkValidControllerRevision(tweakData([]byte(`{"foo":"baz"}`))),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("data"), runtime.RawExtension{Raw: []byte(`{"foo":"baz"}`)}, "field is immutable").WithOrigin("immutable"),
			},
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidControllerRevision(tweaks ...func(*apps.ControllerRevision)) apps.ControllerRevision {
	r := apps.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-revision",
			Namespace: "default",
		},
		Revision: 1,
		Data: runtime.RawExtension{
			Raw: []byte(`{"foo":"bar"}`),
		},
	}
	for _, tweak := range tweaks {
		tweak(&r)
	}
	return r
}

func tweakData(data []byte) func(*apps.ControllerRevision) {
	return func(r *apps.ControllerRevision) {
		r.Data.Raw = data
	}
}
