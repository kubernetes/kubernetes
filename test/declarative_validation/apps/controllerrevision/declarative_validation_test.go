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
	"k8s.io/kubernetes/pkg/registry/apps/controllerrevision"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

// Helper function to create a baseline valid ControllerRevision with optional tweaks
func mkControllerRevision(tweaks ...func(*apps.ControllerRevision)) apps.ControllerRevision {
	obj := apps.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-resource-name",
		},
		Data:     runtime.RawExtension{Raw: []byte(`{"kind":"Foo"}`)},
		Revision: 1,
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := controllerrevision.Strategy
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "apps",
				APIVersion:        apiVersion,
				Resource:          "controllerrevisions",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "create",
			})
			obj := mkControllerRevision(tweakNamespace(namespace))
			meta.RunObjectMetaTestCases(t, ctx, &obj, strategy, meta.WithStringentFinalizerValidation())
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := controllerrevision.Strategy
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "apps",
				APIVersion:        apiVersion,
				Resource:          "controllerrevisions",
				Namespace:         namespace,
				Name:              "valid-obj",
				IsResourceRequest: true,
				Verb:              "update",
			})
			obj := mkControllerRevision(tweakNamespace(namespace))
			meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, strategy, meta.WithStringentFinalizerValidation())

			testCases := map[string]struct {
				old          apps.ControllerRevision
				update       apps.ControllerRevision
				expectedErrs field.ErrorList
			}{
				"valid update": {
					old:    mkControllerRevision(tweakNamespace(namespace)),
					update: mkControllerRevision(tweakNamespace(namespace)),
				},
				"data empty -> some value": {
					old:    mkControllerRevision(tweakNamespace(namespace), tweakData("")),
					update: mkControllerRevision(tweakNamespace(namespace)),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("data"), nil, "").WithOrigin("immutable").MarkAlpha(),
					},
				},
				"data some value -> empty": {
					old:    mkControllerRevision(tweakNamespace(namespace)),
					update: mkControllerRevision(tweakNamespace(namespace), tweakData("")),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("data"), nil, "").WithOrigin("immutable").MarkAlpha(),
					},
				},
				"data some value -> changed value": {
					old:    mkControllerRevision(tweakNamespace(namespace)),
					update: mkControllerRevision(tweakNamespace(namespace), tweakData(`{"kind":"Bar"}`)),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("data"), nil, "").WithOrigin("immutable").MarkAlpha(),
					},
				},
			}
			for k, tc := range testCases {
				t.Run(k, func(t *testing.T) {
					tc.old.ResourceVersion = "1"
					tc.update.ResourceVersion = "2"
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, strategy, tc.expectedErrs)
				})
			}
		})
	}
}

func tweakData(raw string) func(*apps.ControllerRevision) {
	return func(o *apps.ControllerRevision) {
		if raw == "" {
			o.Data = runtime.RawExtension{}
		} else {
			o.Data = runtime.RawExtension{Raw: []byte(raw)}
		}
	}
}

func tweakNamespace(namespace string) func(*apps.ControllerRevision) {
	return func(o *apps.ControllerRevision) {
		o.Namespace = namespace
	}
}
