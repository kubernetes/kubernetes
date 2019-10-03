/*
Copyright 2019 The Kubernetes Authors.

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

package v1beta1_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	v1beta1 "k8s.io/api/admissionregistration/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
	utilpointer "k8s.io/utils/pointer"
)

func TestDefaultAdmissionWebhook(t *testing.T) {
	ignore := v1beta1.Ignore
	exact := v1beta1.Exact
	never := v1beta1.NeverReinvocationPolicy
	thirty := int32(30)
	allScopes := v1beta1.AllScopes
	unknown := v1beta1.SideEffectClassUnknown

	tests := []struct {
		name     string
		original runtime.Object
		expected runtime.Object
	}{
		{
			name: "ValidatingWebhookConfiguration",
			original: &v1beta1.ValidatingWebhookConfiguration{
				Webhooks: []v1beta1.ValidatingWebhook{{}},
			},
			expected: &v1beta1.ValidatingWebhookConfiguration{
				Webhooks: []v1beta1.ValidatingWebhook{{
					FailurePolicy:           &ignore,
					MatchPolicy:             &exact,
					TimeoutSeconds:          &thirty,
					NamespaceSelector:       &metav1.LabelSelector{},
					ObjectSelector:          &metav1.LabelSelector{},
					SideEffects:             &unknown,
					AdmissionReviewVersions: []string{"v1beta1"},
				}},
			},
		},
		{
			name: "MutatingWebhookConfiguration",
			original: &v1beta1.MutatingWebhookConfiguration{
				Webhooks: []v1beta1.MutatingWebhook{{}},
			},
			expected: &v1beta1.MutatingWebhookConfiguration{
				Webhooks: []v1beta1.MutatingWebhook{{
					FailurePolicy:           &ignore,
					MatchPolicy:             &exact,
					ReinvocationPolicy:      &never,
					TimeoutSeconds:          &thirty,
					NamespaceSelector:       &metav1.LabelSelector{},
					ObjectSelector:          &metav1.LabelSelector{},
					SideEffects:             &unknown,
					AdmissionReviewVersions: []string{"v1beta1"},
				}},
			},
		},
		{
			name: "scope=*",
			original: &v1beta1.MutatingWebhookConfiguration{
				Webhooks: []v1beta1.MutatingWebhook{{
					Rules: []v1beta1.RuleWithOperations{{}},
				}},
			},
			expected: &v1beta1.MutatingWebhookConfiguration{
				Webhooks: []v1beta1.MutatingWebhook{{
					Rules: []v1beta1.RuleWithOperations{{Rule: v1beta1.Rule{
						Scope: &allScopes, // defaulted
					}}},
					FailurePolicy:           &ignore,
					MatchPolicy:             &exact,
					ReinvocationPolicy:      &never,
					TimeoutSeconds:          &thirty,
					NamespaceSelector:       &metav1.LabelSelector{},
					ObjectSelector:          &metav1.LabelSelector{},
					SideEffects:             &unknown,
					AdmissionReviewVersions: []string{"v1beta1"},
				}},
			},
		},
		{
			name: "port=443",
			original: &v1beta1.MutatingWebhookConfiguration{
				Webhooks: []v1beta1.MutatingWebhook{{
					ClientConfig: v1beta1.WebhookClientConfig{
						Service: &v1beta1.ServiceReference{},
					},
				}},
			},
			expected: &v1beta1.MutatingWebhookConfiguration{
				Webhooks: []v1beta1.MutatingWebhook{{
					ClientConfig: v1beta1.WebhookClientConfig{
						Service: &v1beta1.ServiceReference{
							Port: utilpointer.Int32Ptr(443), // defaulted
						},
					},
					FailurePolicy:           &ignore,
					MatchPolicy:             &exact,
					ReinvocationPolicy:      &never,
					TimeoutSeconds:          &thirty,
					NamespaceSelector:       &metav1.LabelSelector{},
					ObjectSelector:          &metav1.LabelSelector{},
					SideEffects:             &unknown,
					AdmissionReviewVersions: []string{"v1beta1"},
				}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := test.original
			expected := test.expected
			legacyscheme.Scheme.Default(original)
			if !apiequality.Semantic.DeepEqual(original, expected) {
				t.Error(cmp.Diff(expected, original))
			}
		})
	}
}
