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

package generic

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/namespace"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/object"
)

func TestShouldCallHook(t *testing.T) {
	a := &Webhook{namespaceMatcher: &namespace.Matcher{}, objectMatcher: &object.Matcher{}}

	allScopes := v1.AllScopes
	exactMatch := v1.Exact
	equivalentMatch := v1.Equivalent

	mapper := runtime.NewEquivalentResourceRegistryWithIdentity(func(resource schema.GroupResource) string {
		if resource.Resource == "deployments" {
			// co-locate deployments in all API groups
			return "/deployments"
		}
		return ""
	})
	mapper.RegisterKindFor(schema.GroupVersionResource{"extensions", "v1beta1", "deployments"}, "", schema.GroupVersionKind{"extensions", "v1beta1", "Deployment"})
	mapper.RegisterKindFor(schema.GroupVersionResource{"apps", "v1", "deployments"}, "", schema.GroupVersionKind{"apps", "v1", "Deployment"})
	mapper.RegisterKindFor(schema.GroupVersionResource{"apps", "v1beta1", "deployments"}, "", schema.GroupVersionKind{"apps", "v1beta1", "Deployment"})
	mapper.RegisterKindFor(schema.GroupVersionResource{"apps", "v1alpha1", "deployments"}, "", schema.GroupVersionKind{"apps", "v1alpha1", "Deployment"})

	mapper.RegisterKindFor(schema.GroupVersionResource{"extensions", "v1beta1", "deployments"}, "scale", schema.GroupVersionKind{"extensions", "v1beta1", "Scale"})
	mapper.RegisterKindFor(schema.GroupVersionResource{"apps", "v1", "deployments"}, "scale", schema.GroupVersionKind{"autoscaling", "v1", "Scale"})
	mapper.RegisterKindFor(schema.GroupVersionResource{"apps", "v1beta1", "deployments"}, "scale", schema.GroupVersionKind{"apps", "v1beta1", "Scale"})
	mapper.RegisterKindFor(schema.GroupVersionResource{"apps", "v1alpha1", "deployments"}, "scale", schema.GroupVersionKind{"apps", "v1alpha1", "Scale"})

	// register invalid kinds to trigger an error
	mapper.RegisterKindFor(schema.GroupVersionResource{"example.com", "v1", "widgets"}, "", schema.GroupVersionKind{"", "", ""})
	mapper.RegisterKindFor(schema.GroupVersionResource{"example.com", "v2", "widgets"}, "", schema.GroupVersionKind{"", "", ""})

	interfaces := &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: mapper}

	testcases := []struct {
		name string

		webhook *v1.ValidatingWebhook
		attrs   admission.Attributes

		expectCall            bool
		expectErr             string
		expectCallResource    schema.GroupVersionResource
		expectCallSubresource string
		expectCallKind        schema.GroupVersionKind
	}{
		{
			name:       "no rules (just write)",
			webhook:    &v1.ValidatingWebhook{Rules: []v1.RuleWithOperations{}},
			attrs:      admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"apps", "v1", "Deployment"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall: false,
		},
		{
			name: "invalid kind lookup",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				MatchPolicy:       &equivalentMatch,
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"example.com"}, APIVersions: []string{"v1"}, Resources: []string{"widgets"}, Scope: &allScopes},
				}}},
			attrs:      admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"example.com", "v2", "Widget"}, "ns", "name", schema.GroupVersionResource{"example.com", "v2", "widgets"}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall: false,
			expectErr:  "unknown kind",
		},
		{
			name: "wildcard rule, match as requested",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*"}, Scope: &allScopes},
				}}},
			attrs:                 admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"apps", "v1", "Deployment"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        schema.GroupVersionKind{"apps", "v1", "Deployment"},
			expectCallResource:    schema.GroupVersionResource{"apps", "v1", "deployments"},
			expectCallSubresource: "",
		},
		{
			name: "specific rules, prefer exact match",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}}},
			attrs:                 admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"apps", "v1", "Deployment"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        schema.GroupVersionKind{"apps", "v1", "Deployment"},
			expectCallResource:    schema.GroupVersionResource{"apps", "v1", "deployments"},
			expectCallSubresource: "",
		},
		{
			name: "specific rules, match miss",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}}},
			attrs:      admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"apps", "v1", "Deployment"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall: false,
		},
		{
			name: "specific rules, exact match miss",
			webhook: &v1.ValidatingWebhook{
				MatchPolicy:       &exactMatch,
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}}},
			attrs:      admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"apps", "v1", "Deployment"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall: false,
		},
		{
			name: "specific rules, equivalent match, prefer extensions",
			webhook: &v1.ValidatingWebhook{
				MatchPolicy:       &equivalentMatch,
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}}},
			attrs:                 admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"apps", "v1", "Deployment"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        schema.GroupVersionKind{"extensions", "v1beta1", "Deployment"},
			expectCallResource:    schema.GroupVersionResource{"extensions", "v1beta1", "deployments"},
			expectCallSubresource: "",
		},
		{
			name: "specific rules, equivalent match, prefer apps",
			webhook: &v1.ValidatingWebhook{
				MatchPolicy:       &equivalentMatch,
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments"}, Scope: &allScopes},
				}}},
			attrs:                 admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"apps", "v1", "Deployment"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        schema.GroupVersionKind{"apps", "v1beta1", "Deployment"},
			expectCallResource:    schema.GroupVersionResource{"apps", "v1beta1", "deployments"},
			expectCallSubresource: "",
		},

		{
			name: "specific rules, subresource prefer exact match",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}}},
			attrs:                 admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"autoscaling", "v1", "Scale"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        schema.GroupVersionKind{"autoscaling", "v1", "Scale"},
			expectCallResource:    schema.GroupVersionResource{"apps", "v1", "deployments"},
			expectCallSubresource: "scale",
		},
		{
			name: "specific rules, subresource match miss",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}}},
			attrs:      admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"autoscaling", "v1", "Scale"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall: false,
		},
		{
			name: "specific rules, subresource exact match miss",
			webhook: &v1.ValidatingWebhook{
				MatchPolicy:       &exactMatch,
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}}},
			attrs:      admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"autoscaling", "v1", "Scale"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall: false,
		},
		{
			name: "specific rules, subresource equivalent match, prefer extensions",
			webhook: &v1.ValidatingWebhook{
				MatchPolicy:       &equivalentMatch,
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}}},
			attrs:                 admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"autoscaling", "v1", "Scale"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        schema.GroupVersionKind{"extensions", "v1beta1", "Scale"},
			expectCallResource:    schema.GroupVersionResource{"extensions", "v1beta1", "deployments"},
			expectCallSubresource: "scale",
		},
		{
			name: "specific rules, subresource equivalent match, prefer apps",
			webhook: &v1.ValidatingWebhook{
				MatchPolicy:       &equivalentMatch,
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}, {
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
				}}},
			attrs:                 admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{"autoscaling", "v1", "Scale"}, "ns", "name", schema.GroupVersionResource{"apps", "v1", "deployments"}, "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        schema.GroupVersionKind{"apps", "v1beta1", "Scale"},
			expectCallResource:    schema.GroupVersionResource{"apps", "v1beta1", "deployments"},
			expectCallSubresource: "scale",
		},
	}

	for i, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			invocation, err := a.ShouldCallHook(webhook.NewValidatingWebhookAccessor(fmt.Sprintf("webhook-%d", i), fmt.Sprintf("webhook-cfg-%d", i), testcase.webhook), testcase.attrs, interfaces)
			if err != nil {
				if len(testcase.expectErr) == 0 {
					t.Fatal(err)
				}
				if !strings.Contains(err.Error(), testcase.expectErr) {
					t.Fatalf("expected error containing %q, got %s", testcase.expectErr, err.Error())
				}
				return
			} else if len(testcase.expectErr) > 0 {
				t.Fatalf("expected error %q, got no error and %#v", testcase.expectErr, invocation)
			}

			if invocation == nil {
				if testcase.expectCall {
					t.Fatal("expected invocation, got nil")
				}
				return
			}

			if !testcase.expectCall {
				t.Fatal("unexpected invocation")
			}

			if invocation.Kind != testcase.expectCallKind {
				t.Fatalf("expected %#v, got %#v", testcase.expectCallKind, invocation.Kind)
			}
			if invocation.Resource != testcase.expectCallResource {
				t.Fatalf("expected %#v, got %#v", testcase.expectCallResource, invocation.Resource)
			}
			if invocation.Subresource != testcase.expectCallSubresource {
				t.Fatalf("expected %#v, got %#v", testcase.expectCallSubresource, invocation.Subresource)
			}
		})
	}
}
