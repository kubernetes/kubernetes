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

package persistentvolumeclaim

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	core "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/persistentvolumeclaim"
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
		Resource:          "persistentvolumeclaims",
		IsResourceRequest: true,
		Verb:              "create",
	}), metav1.NamespaceDefault)

	obj := mkValidPersistentVolumeClaim()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "api",
		APIGroup:          "",
		APIVersion:        apiVersion,
		Resource:          "persistentvolumeclaims",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	}), metav1.NamespaceDefault)

	updateObj := mkValidPersistentVolumeClaim()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "api",
			APIGroup:          "",
			APIVersion:        apiVersion,
			Resource:          "persistentvolumeclaims",
			Subresource:       "status",
			Name:              "valid-obj",
			IsResourceRequest: true,
			Verb:              "update",
		}), metav1.NamespaceDefault)

		tests := map[string]struct {
			conditions   []core.VolumeHealthCondition
			expectedErrs field.ErrorList
		}{
			"valid": {
				conditions: []core.VolumeHealthCondition{{Status: core.VolumeHealthDegraded, Reason: "DiskSlow"}},
			},
			"status required": {
				conditions:   []core.VolumeHealthCondition{{Reason: "DiskSlow"}},
				expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "healthStatus", "healthConditions").Index(0).Child("status"), "")},
			},
			"status enum": {
				conditions:   []core.VolumeHealthCondition{{Status: "Invalid", Reason: "DiskSlow"}},
				expectedErrs: field.ErrorList{field.NotSupported(field.NewPath("status", "healthStatus", "healthConditions").Index(0).Child("status"), core.VolumeHealthStatusType("Invalid"), []core.VolumeHealthStatusType(nil))},
			},
			"reason required": {
				conditions:   []core.VolumeHealthCondition{{Status: core.VolumeHealthDegraded}},
				expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "healthStatus", "healthConditions").Index(0).Child("reason"), "")},
			},
			"reason max bytes": {
				conditions:   []core.VolumeHealthCondition{{Status: core.VolumeHealthDegraded, Reason: strings.Repeat("a", 257)}},
				expectedErrs: field.ErrorList{field.TooLong(field.NewPath("status", "healthStatus", "healthConditions").Index(0).Child("reason"), "", 256).WithOrigin("maxBytes")},
			},
			"message max bytes": {
				conditions:   []core.VolumeHealthCondition{{Status: core.VolumeHealthDegraded, Reason: "DiskSlow", Message: strings.Repeat("𝄞", 257)}},
				expectedErrs: field.ErrorList{field.TooLong(field.NewPath("status", "healthStatus", "healthConditions").Index(0).Child("message"), "", 1024).WithOrigin("maxBytes")},
			},
			"duplicate conditions": {
				conditions: []core.VolumeHealthCondition{
					{Status: core.VolumeHealthDegraded, Reason: "DiskSlow"},
					{Status: core.VolumeHealthDegraded, Reason: "DiskSlow"},
				},
				expectedErrs: field.ErrorList{field.Duplicate(field.NewPath("status", "healthStatus", "healthConditions").Index(1), nil)},
			},
			"too many conditions": {
				conditions:   makeVolumeHealthConditions(17),
				expectedErrs: field.ErrorList{field.TooMany(field.NewPath("status", "healthStatus", "healthConditions"), 17, 16).WithOrigin("maxItems")},
			},
		}

		for name, tc := range tests {
			t.Run(apiVersion+"/"+name, func(t *testing.T) {
				oldObj := mkValidPersistentVolumeClaim()
				oldObj.ResourceVersion = "1"
				updateObj := oldObj.DeepCopy()
				updateObj.Status.HealthStatus = &core.VolumeHealthStatus{HealthConditions: tc.conditions}
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, updateObj, &oldObj, registry.StatusStrategy, tc.expectedErrs, apitesting.WithSubResources("status"))
			})
		}
	}
}

func makeVolumeHealthConditions(count int) []core.VolumeHealthCondition {
	conditions := make([]core.VolumeHealthCondition, count)
	for i := range conditions {
		conditions[i] = core.VolumeHealthCondition{
			Status: core.VolumeHealthDegraded,
			Reason: "Reason" + string(rune('A'+i)),
		}
	}
	return conditions
}

func mkValidPersistentVolumeClaim() core.PersistentVolumeClaim {
	return core.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-obj",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: core.PersistentVolumeClaimSpec{
			AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
			Resources: core.VolumeResourceRequirements{
				Requests: core.ResourceList{core.ResourceStorage: resource.MustParse("1Gi")},
			},
		},
	}
}
