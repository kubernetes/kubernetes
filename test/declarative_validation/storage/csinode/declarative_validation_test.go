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

package csinode

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	storage "k8s.io/kubernetes/pkg/apis/storage"
	registry "k8s.io/kubernetes/pkg/registry/storage/csinode"
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
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "csinodes",
		IsResourceRequest: true,
		Verb:              "create",
	})

	obj := mkCSINode()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "csinodes",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})

	updateObj := mkCSINode()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy,
		meta.WithStringentFinalizerValidation())
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "apis",
			APIGroup:          "storage.k8s.io",
			APIVersion:        apiVersion,
			Resource:          "csinodes",
			Subresource:       "status",
			Name:              "valid-obj",
			IsResourceRequest: true,
			Verb:              "update",
		})

		tests := map[string]struct {
			storageHealth []storage.StorageHealth
			expectedErrs  field.ErrorList
		}{
			"valid": {
				storageHealth: []storage.StorageHealth{{Name: "foo"}},
			},
			"name required": {
				storageHealth: []storage.StorageHealth{{}},
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "storageHealth").Index(0).Child("name"), "").MarkAlpha(),
				},
			},
		}

		for name, tc := range tests {
			t.Run(apiVersion+"/"+name, func(t *testing.T) {
				oldObj := mkCSINode()
				oldObj.ResourceVersion = "1"
				updateObj := oldObj.DeepCopy()
				updateObj.Status.StorageHealth = tc.storageHealth
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, updateObj, &oldObj,
					registry.StatusStrategy, tc.expectedErrs,
					apitesting.WithSubResources("status"))
			})
		}
	}
}

func mkCSINode(tweaks ...func(node *storage.CSINode)) storage.CSINode {
	node := storage.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-obj",
		},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{
					Name:   "foo",
					NodeID: "bar",
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&node)
	}
	return node
}
