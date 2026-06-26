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

package poddisruptionbudget

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/policy"
	registry "k8s.io/kubernetes/pkg/registry/policy/poddisruptionbudget"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "policy",
				APIVersion:        apiVersion,
				Resource:          "poddisruptionbudgets",
				IsResourceRequest: true,
				Verb:              "create",
			}), metav1.NamespaceDefault)

			meta.RunObjectMetaTestCases(t, ctx, &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "valid-pdb",
					Namespace: metav1.NamespaceDefault,
				},
			}, registry.Strategy)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "policy",
				APIVersion:        apiVersion,
				Resource:          "poddisruptionbudgets",
				IsResourceRequest: true,
				Verb:              "update",
			}), metav1.NamespaceDefault)

			meta.RunObjectMetaUpdateTestCases(t, ctx, &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "valid-pdb",
					Namespace: metav1.NamespaceDefault,
				},
			}, registry.Strategy)
		})
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "policy",
				APIVersion:        apiVersion,
				Resource:          "poddisruptionbudgets",
				Subresource:       "status",
				IsResourceRequest: true,
				Verb:              "update",
			})

			meta.RunConditionTestCases(t, ctx, field.NewPath("status", "conditions"), &policy.PodDisruptionBudget{}, registry.StatusStrategy, func(obj *policy.PodDisruptionBudget, c []metav1.Condition) {
				*obj = policy.PodDisruptionBudget{
					ObjectMeta: metav1.ObjectMeta{Name: "valid-pdb", Namespace: "default", ResourceVersion: "1"},
					Spec:       policy.PodDisruptionBudgetSpec{},
					Status: policy.PodDisruptionBudgetStatus{
						Conditions: c,
					},
				}
			})
		})
	}
}
