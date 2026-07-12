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

package controllerrevision

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	apps "k8s.io/kubernetes/pkg/apis/apps"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	registry "k8s.io/kubernetes/pkg/registry/apps/controllerrevision"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidate(t, apiVersion)
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "apps",
		APIVersion:        apiVersion,
		IsResourceRequest: true,
		Verb:              "create",
	})
	testCases := map[string]struct {
		input        apps.ControllerRevision
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidControllerRevision(),
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}

	obj := mkValidControllerRevision()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateUpdate(t, apiVersion)
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "apps",
		APIVersion:        apiVersion,
		IsResourceRequest: true,
		Verb:              "update",
	})
	testCases := map[string]struct {
		old          apps.ControllerRevision
		update       apps.ControllerRevision
		expectedErrs field.ErrorList
	}{
		"valid update": {
			old:    mkValidControllerRevision(),
			update: mkValidControllerRevision(),
		},
		"changed data": {
			old:    mkValidControllerRevision(),
			update: mkValidControllerRevision(tweakData(runtime.RawExtension{})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("data"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, registry.Strategy, tc.expectedErrs)
		})
	}

	updateObj := mkValidControllerRevision()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkValidControllerRevision(tweaks ...func(*apps.ControllerRevision)) apps.ControllerRevision {
	obj := apps.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-resource-name",
			Namespace: metav1.NamespaceDefault,
		},
		Data:     runtime.RawExtension{Raw: []byte(`{"kind":"Foo"}`)},
		Revision: 1,
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func tweakData(data runtime.RawExtension) func(*apps.ControllerRevision) {
	return func(o *apps.ControllerRevision) {
		o.Data = data
	}
}
