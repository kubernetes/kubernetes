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

package storageversion

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/apiserverinternal"
	_ "k8s.io/kubernetes/pkg/apis/apiserverinternal/install"
	registry "k8s.io/kubernetes/pkg/registry/apiserverinternal/storageversion"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

// TODO: remove this apiVersions variable once coverage tests are generated for this package.
var apiVersions = []string{"v1alpha1"}

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
		APIGroup:          "internal.apiserver.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "storageversions",
		IsResourceRequest: true,
		Verb:              "create",
	})

	obj := mkValidStorageVersion()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "internal.apiserver.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "storageversions",
		Name:              "group.resource",
		IsResourceRequest: true,
		Verb:              "update",
	})

	updateObj := mkValidStorageVersion()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy)
}

func mkValidStorageVersion() apiserverinternal.StorageVersion {
	return apiserverinternal.StorageVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name: "group.resource",
		},
	}
}
