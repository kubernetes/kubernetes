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

package validatingadmissionpolicy

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	registry "k8s.io/kubernetes/pkg/registry/admissionregistration/validatingadmissionpolicy"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:    "admissionregistration.k8s.io",
				APIVersion:  apiVersion,
				Resource:    "validatingadmissionpolicies",
				Subresource: "status",
			})

			strategy := registry.NewStatusStrategy(registry.NewStrategy(nil, nil))
			meta.RunConditionTestCases(t, ctx, field.NewPath("status", "conditions"), &admissionregistration.ValidatingAdmissionPolicy{}, strategy, func(obj *admissionregistration.ValidatingAdmissionPolicy, c []metav1.Condition) {
				*obj = admissionregistration.ValidatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "valid-policy", ResourceVersion: "1"},
					Spec:       admissionregistration.ValidatingAdmissionPolicySpec{},
					Status: admissionregistration.ValidatingAdmissionPolicyStatus{
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
					obj := &admissionregistration.ValidatingAdmissionPolicy{
						ObjectMeta: metav1.ObjectMeta{Name: "valid-policy", ResourceVersion: "1"},
						Spec:       admissionregistration.ValidatingAdmissionPolicySpec{},
						Status: admissionregistration.ValidatingAdmissionPolicyStatus{
							Conditions: tc.Conditions,
						},
					}
					old := &admissionregistration.ValidatingAdmissionPolicy{
						ObjectMeta: metav1.ObjectMeta{Name: "valid-policy", ResourceVersion: "1"},
					}
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, obj, old, strategy, tc.ExpectedErrs)
				})
			}
		})
	}
}

// Helper function to create a baseline valid ValidatingAdmissionPolicy with optional tweaks
func mkValidatingAdmissionPolicy(tweaks ...func(*admissionregistration.ValidatingAdmissionPolicy)) admissionregistration.ValidatingAdmissionPolicy {
	fp := admissionregistration.Fail
	mp := admissionregistration.Equivalent
	obj := admissionregistration.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-resource-name",
		},
		Spec: admissionregistration.ValidatingAdmissionPolicySpec{
			FailurePolicy: &fp,
			MatchConstraints: &admissionregistration.MatchResources{
				MatchPolicy:       &mp,
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				ResourceRules: []admissionregistration.NamedRuleWithOperations{
					{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{
								admissionregistration.Create,
							},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"*"},
								APIVersions: []string{"*"},
								Resources:   []string{"*"},
							},
						},
					},
				},
			},
			Validations: []admissionregistration.Validation{
				{
					Expression: "true",
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := registry.NewStrategy(nil, nil)
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "admissionregistration.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "validatingadmissionpolicies",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "create",
			})
			obj := mkValidatingAdmissionPolicy(func(o *admissionregistration.ValidatingAdmissionPolicy) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaTestCases(t, ctx, &obj, strategy)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := registry.NewStrategy(nil, nil)
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "admissionregistration.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "validatingadmissionpolicies",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "update",
			})
			obj := mkValidatingAdmissionPolicy(func(o *admissionregistration.ValidatingAdmissionPolicy) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, strategy)
		})
	}
}
