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
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/policy"
	registry "k8s.io/kubernetes/pkg/registry/policy/poddisruptionbudget"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:    "policy",
				APIVersion:  apiVersion,
				Resource:    "poddisruptionbudgets",
				Subresource: "status",
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
			testCases := []meta.ConditionTestCase{
				{
					Name: "invalid type format not a k8s label key",
					Conditions: []metav1.Condition{
						meta.MkCondition(
							meta.TweakType("INVALID TYPE"),
						),
					},
					ExpectedErrs: field.ErrorList{
						field.Invalid(
							field.NewPath("status", "conditions").Index(0).Child("type"),
							"INVALID TYPE",
							"name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')",
						).WithOrigin("format=k8s-label-key").MarkAlpha(),
					},
				},
			}
			for _, tc := range testCases {
				t.Run("conditions: "+tc.Name, func(t *testing.T) {
					obj := &policy.PodDisruptionBudget{
						ObjectMeta: metav1.ObjectMeta{Name: "valid-pdb", Namespace: "default", ResourceVersion: "1"},
						Spec:       policy.PodDisruptionBudgetSpec{},
						Status: policy.PodDisruptionBudgetStatus{
							Conditions: tc.Conditions,
						},
					}
					old := &policy.PodDisruptionBudget{
						ObjectMeta: metav1.ObjectMeta{Name: "valid-pdb", Namespace: "default", ResourceVersion: "1"},
					}
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, obj, old, registry.StatusStrategy, tc.ExpectedErrs)
				})
			}
		})
	}
}
