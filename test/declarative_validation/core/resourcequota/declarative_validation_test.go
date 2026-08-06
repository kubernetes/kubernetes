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

package resourcequota

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	core "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/resourcequota"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "api",
		APIGroup:          "",
		APIVersion:        apiVersion,
		Resource:          "resourcequotas",
		IsResourceRequest: true,
		Verb:              "create",
	}), metav1.NamespaceDefault)

	obj := mkValidResourceQuota()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "api",
		APIGroup:          "",
		APIVersion:        apiVersion,
		Resource:          "resourcequotas",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	}), metav1.NamespaceDefault)

	updateObj := mkValidResourceQuota()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())

	testCases := map[string]struct {
		oldObj       core.ResourceQuota
		updateObj    core.ResourceQuota
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidResourceQuota(tweakScopes(core.ResourceQuotaScopeBestEffort)),
			updateObj: mkValidResourceQuota(tweakScopes(core.ResourceQuotaScopeBestEffort)),
		},
		"scopes changed": {
			oldObj:    mkValidResourceQuota(tweakScopes(core.ResourceQuotaScopeBestEffort)),
			updateObj: mkValidResourceQuota(tweakScopes(core.ResourceQuotaScopeNotBestEffort)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("scopes"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"scopes set from unset": {
			oldObj:    mkValidResourceQuota(),
			updateObj: mkValidResourceQuota(tweakScopes(core.ResourceQuotaScopeBestEffort)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("scopes"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"scopes unset from set": {
			oldObj:    mkValidResourceQuota(tweakScopes(core.ResourceQuotaScopeBestEffort)),
			updateObj: mkValidResourceQuota(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("scopes"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}
}

func mkValidResourceQuota(tweaks ...func(*core.ResourceQuota)) core.ResourceQuota {
	obj := core.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-obj",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: core.ResourceQuotaSpec{},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func tweakScopes(scopes ...core.ResourceQuotaScope) func(*core.ResourceQuota) {
	return func(obj *core.ResourceQuota) {
		obj.Spec.Scopes = scopes
	}
}
