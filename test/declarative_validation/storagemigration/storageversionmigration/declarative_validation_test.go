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

package storageversionmigration

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/storagemigration"
	_ "k8s.io/kubernetes/pkg/apis/storagemigration/install"
	registry "k8s.io/kubernetes/pkg/registry/storagemigration/storagemigration"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "storagemigration.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "storageversionmigrations",
				IsResourceRequest: true,
				Verb:              "create",
			})

			meta.RunObjectMetaTestCases(t, ctx, &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "valid-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: metav1.GroupResource{
						Group:    "group",
						Resource: "resources",
					},
				},
			}, registry.Strategy, meta.WithStringentFinalizerValidation())
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "storagemigration.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "storageversionmigrations",
				IsResourceRequest: true,
				Verb:              "update",
			})

			meta.RunObjectMetaUpdateTestCases(t, ctx, &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "valid-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: metav1.GroupResource{
						Group:    "group",
						Resource: "resources",
					},
				},
			}, registry.Strategy, meta.WithStringentFinalizerValidation())
		})
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "storagemigration.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "storageversionmigrations",
				Subresource:       "status",
				IsResourceRequest: true,
				Verb:              "update",
			})

			meta.RunConditionTestCases(t, ctx, field.NewPath("status", "conditions"), &storagemigration.StorageVersionMigration{}, registry.StatusStrategy, func(obj *storagemigration.StorageVersionMigration, c []metav1.Condition) {
				*obj = storagemigration.StorageVersionMigration{
					ObjectMeta: metav1.ObjectMeta{Name: "valid-svm", ResourceVersion: "1"},
					Spec:       storagemigration.StorageVersionMigrationSpec{},
					Status: storagemigration.StorageVersionMigrationStatus{
						Conditions: c,
					},
				}
			})
		})
	}
}
