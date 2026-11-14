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
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/coverage"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/matchconditions"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/namespace"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/object"
	webhooktesting "k8s.io/apiserver/pkg/admission/plugin/webhook/testing"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func gvr(group, version, resource string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
}

func gvk(group, version, kind string) schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: group, Version: version, Kind: kind}
}

var _ matchconditions.Matcher = &fakeMatcher{}

type fakeMatcher struct {
	throwError  error
	matchResult bool
}

func (f *fakeMatcher) Match(ctx context.Context, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, authz authorizer.UnconditionalAuthorizer) matchconditions.MatchResult {
	if f.throwError != nil {
		return matchconditions.MatchResult{
			Matches:             true,
			FailedConditionName: "",
			Error:               f.throwError,
		}
	}
	return matchconditions.MatchResult{
		Matches:             f.matchResult,
		FailedConditionName: "",
	}
}

var _ webhook.WebhookAccessor = &fakeWebhookAccessor{}

type fakeWebhookAccessor struct {
	webhook.WebhookAccessor
	throwError  error
	matchResult bool
}

func (f *fakeWebhookAccessor) GetCompiledMatcher(compiler cel.ConditionCompiler) matchconditions.Matcher {
	return &fakeMatcher{
		throwError:  f.throwError,
		matchResult: f.matchResult,
	}
}

var _ VersionedAttributeAccessor = &fakeVersionedAttributeAccessor{}

type fakeVersionedAttributeAccessor struct{}

func (v *fakeVersionedAttributeAccessor) VersionedAttribute(gvk schema.GroupVersionKind) (*admission.VersionedAttributes, error) {
	return nil, nil
}

func TestShouldCallHook(t *testing.T) {
	a := &Webhook{
		namespaceMatcher: &namespace.Matcher{},
		objectMatcher:    &object.Matcher{},
	}

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
	mapper.RegisterKindFor(gvr("extensions", "v1beta1", "deployments"), "", gvk("extensions", "v1beta1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1", "deployments"), "", gvk("apps", "v1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "deployments"), "", gvk("apps", "v1beta1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha1", "deployments"), "", gvk("apps", "v1alpha1", "Deployment"))

	mapper.RegisterKindFor(gvr("extensions", "v1beta1", "deployments"), "scale", gvk("extensions", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1", "deployments"), "scale", gvk("autoscaling", "v1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "deployments"), "scale", gvk("apps", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha1", "deployments"), "scale", gvk("apps", "v1alpha1", "Scale"))

	// register invalid kinds to trigger an error
	mapper.RegisterKindFor(gvr("example.com", "v1", "widgets"), "", gvk("", "", ""))
	mapper.RegisterKindFor(gvr("example.com", "v2", "widgets"), "", gvk("", "", ""))

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
		matchError            error
		matchResult           bool
	}{
		{
			name:        "no rules (just write)",
			webhook:     &v1.ValidatingWebhook{NamespaceSelector: &metav1.LabelSelector{}, Rules: []v1.RuleWithOperations{}},
			attrs:       admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:  false,
			matchResult: true,
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
			attrs:       admission.NewAttributesRecord(nil, nil, gvk("example.com", "v2", "Widget"), "ns", "name", gvr("example.com", "v2", "widgets"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:  false,
			expectErr:   "unknown kind",
			matchResult: true,
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
			attrs:                 admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        gvk("apps", "v1", "Deployment"),
			expectCallResource:    gvr("apps", "v1", "deployments"),
			expectCallSubresource: "",
			matchResult:           true,
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
			attrs:                 admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        gvk("apps", "v1", "Deployment"),
			expectCallResource:    gvr("apps", "v1", "deployments"),
			expectCallSubresource: "",
			matchResult:           true,
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
			attrs:       admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:  false,
			matchResult: true,
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
			attrs:       admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:  false,
			matchResult: true,
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
			attrs:                 admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        gvk("extensions", "v1beta1", "Deployment"),
			expectCallResource:    gvr("extensions", "v1beta1", "deployments"),
			expectCallSubresource: "",
			matchResult:           true,
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
			attrs:                 admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        gvk("apps", "v1beta1", "Deployment"),
			expectCallResource:    gvr("apps", "v1beta1", "deployments"),
			expectCallSubresource: "",
			matchResult:           true,
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
			attrs:                 admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "ns", "name", gvr("apps", "v1", "deployments"), "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        gvk("autoscaling", "v1", "Scale"),
			expectCallResource:    gvr("apps", "v1", "deployments"),
			expectCallSubresource: "scale",
			matchResult:           true,
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
			attrs:       admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "ns", "name", gvr("apps", "v1", "deployments"), "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:  false,
			matchResult: true,
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
			attrs:       admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "ns", "name", gvr("apps", "v1", "deployments"), "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:  false,
			matchResult: true,
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
			attrs:                 admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "ns", "name", gvr("apps", "v1", "deployments"), "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        gvk("extensions", "v1beta1", "Scale"),
			expectCallResource:    gvr("extensions", "v1beta1", "deployments"),
			expectCallSubresource: "scale",
			matchResult:           true,
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
			attrs:                 admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "ns", "name", gvr("apps", "v1", "deployments"), "scale", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        gvk("apps", "v1beta1", "Scale"),
			expectCallResource:    gvr("apps", "v1beta1", "deployments"),
			expectCallSubresource: "scale",
			matchResult:           true,
		},
		{
			name: "wildcard rule, match conditions also match",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*"}, Scope: &allScopes},
				}},
				MatchConditions: []v1.MatchCondition{
					{
						Name:       "test1",
						Expression: "test expression",
					},
				},
			},
			attrs:                 admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            true,
			expectCallKind:        gvk("apps", "v1", "Deployment"),
			expectCallResource:    gvr("apps", "v1", "deployments"),
			expectCallSubresource: "",
			matchResult:           true,
		},
		{
			name: "wildcard rule, match conditions do not match",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*"}, Scope: &allScopes},
				}},
				MatchConditions: []v1.MatchCondition{
					{
						Name:       "test1",
						Expression: "test expression",
					},
				},
			},
			attrs:       admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:  false,
			matchResult: false,
		},
		{
			name: "wildcard rule, match conditions error",
			webhook: &v1.ValidatingWebhook{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				Rules: []v1.RuleWithOperations{{
					Operations: []v1.OperationType{"*"},
					Rule:       v1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*"}, Scope: &allScopes},
				}},
				MatchConditions: []v1.MatchCondition{
					{
						Name:       "test1",
						Expression: "test expression",
					},
				},
			},
			attrs:                 admission.NewAttributesRecord(nil, nil, gvk("apps", "v1", "Deployment"), "ns", "name", gvr("apps", "v1", "deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:            false,
			expectErr:             "deployments.apps \"name\" is forbidden: test error",
			expectCallKind:        gvk("apps", "v1", "Deployment"),
			expectCallResource:    gvr("apps", "v1", "deployments"),
			expectCallSubresource: "",
			matchError:            errors.New("test error"),
		},
	}

	for i, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			fakeWebhook := &fakeWebhookAccessor{
				WebhookAccessor: webhook.NewValidatingWebhookAccessor(fmt.Sprintf("webhook-%d", i), fmt.Sprintf("webhook-cfg-%d", i), testcase.webhook),
				matchResult:     testcase.matchResult,
				throwError:      testcase.matchError,
			}

			invocation, err := a.ShouldCallHook(context.TODO(), fakeWebhook, testcase.attrs, interfaces, &fakeVersionedAttributeAccessor{})
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

type fakeNamespaceLister struct {
	namespaces map[string]*corev1.Namespace
}

func (f fakeNamespaceLister) List(selector labels.Selector) (ret []*corev1.Namespace, err error) {
	return nil, nil
}
func (f fakeNamespaceLister) Get(name string) (*corev1.Namespace, error) {
	ns, ok := f.namespaces[name]
	if ok {
		return ns, nil
	}
	return nil, k8serrors.NewNotFound(corev1.Resource("namespaces"), name)
}

func podRule(ops []v1.OperationType, resources ...string) v1.RuleWithOperations {
	return v1.RuleWithOperations{
		Operations: ops,
		Rule:       v1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: resources},
	}
}

func TestShouldCallHookCoverage(t *testing.T) {
	createUpdate := []v1.OperationType{v1.Create, v1.Update}
	update := []v1.OperationType{v1.Update}
	matchConditions := []v1.MatchCondition{{Name: "c", Expression: "unused: fake matcher"}}
	badSelector := &metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{{Key: "k", Operator: "BadOp"}}}

	dynamic := webhooktesting.NewCoverageRequirements(t)
	mainResource := []coverage.Requirement{{
		Resource: schema.GroupResource{Resource: "pods"},
		Target:   coverage.Shape{Operation: admission.Update},
		Sources:  []coverage.Shape{{Operation: admission.Create}},
	}}

	target := func() *webhooktesting.FakeAttributes {
		return webhooktesting.NewPodAttribute(admission.Update, "dynamic", map[string]string{"app": "y"})
	}
	interfaces := &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: runtime.NewEquivalentResourceRegistry()}
	namespaceLister := fakeNamespaceLister{map[string]*corev1.Namespace{
		"default": {ObjectMeta: metav1.ObjectMeta{Name: "default", Labels: map[string]string{"env": "prod"}}},
	}}
	annotationKey := coverage.AuditAnnotationKey(coverage.WebhookValidating)

	// apps and extensions deployments are equivalent, so that a hook can match
	// an apps request only through matchPolicy: Equivalent.
	equivalentMapper := runtime.NewEquivalentResourceRegistryWithIdentity(func(resource schema.GroupResource) string {
		if resource.Resource == "deployments" {
			return "/deployments"
		}
		return ""
	})
	equivalentMapper.RegisterKindFor(gvr("apps", "v1", "deployments"), "scale", gvk("autoscaling", "v1", "Scale"))
	equivalentMapper.RegisterKindFor(gvr("extensions", "v1beta1", "deployments"), "scale", gvk("extensions", "v1beta1", "Scale"))
	deploymentScale := []coverage.Requirement{{
		Resource: schema.GroupResource{Group: "apps", Resource: "deployments"},
		Target:   coverage.Shape{Operation: admission.Update, Subresource: "scale"},
		Sources:  []coverage.Shape{{Operation: admission.Create}, {Operation: admission.Update}},
	}}
	// Intercepts apps deployments, but their scale subresource only in extensions.
	equivalentScaleRules := []v1.RuleWithOperations{{
		Operations: createUpdate,
		Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1"}, Resources: []string{"deployments"}},
	}, {
		Operations: update,
		Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments/scale"}},
	}}
	deploymentScaleTarget := func() *webhooktesting.FakeAttributes {
		return &webhooktesting.FakeAttributes{Attributes: admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "default", "d",
			gvr("apps", "v1", "deployments"), "scale", admission.Update, &metav1.UpdateOptions{}, false, nil)}
	}
	equivalent, exact := v1.Equivalent, v1.Exact

	testcases := []struct {
		name              string
		reqs              []coverage.Requirement
		rules             []v1.RuleWithOperations
		matchPolicy       *v1.MatchPolicyType
		mapper            runtime.EquivalentResourceMapper
		namespaceSelector *metav1.LabelSelector // nil: match everything
		objectSelector    *metav1.LabelSelector // nil: match everything
		matchConditions   []v1.MatchCondition
		matchResult       bool
		matchError        error
		attrs             *webhooktesting.FakeAttributes
		reinvoke          bool
		versioned         bool // pass attrs as the mutating dispatcher does after a mutation

		expectCall bool
		expectCode int32    // 0: no error
		expectMsg  []string // substrings of the error message
		rejectMsg  []string // substrings that must NOT appear
		// expectAnnotation, if set, is the exact coverage audit annotation.
		// Every 403 must be annotated and nothing else may be.
		expectAnnotation string
	}{
		// Violations.
		{
			name: "violation: intercepts CREATE,UPDATE pods but not pods/dynamic", reqs: dynamic,
			rules: []v1.RuleWithOperations{podRule(createUpdate, "pods")}, attrs: target(),
			expectCode: 403,
			expectMsg: []string{
				`pods "p" is forbidden: admission webhook "hook.example.com" (configuration "cfg") intercepts CREATE pods but not UPDATE pods/dynamic`,
				`; add a rule for UPDATE on "pods/dynamic" (or "pods/*") to the webhook's rules to allow this request`,
			},
			expectAnnotation: `[{"name":"hook.example.com","configuration":"cfg","source":"CREATE pods","target":"UPDATE pods/dynamic"}]`,
		},
		{
			name: "violation: matching namespace selector is honored, not skipped", reqs: dynamic,
			rules:             []v1.RuleWithOperations{podRule(createUpdate, "pods")},
			namespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "prod"}},
			attrs:             target(), expectCode: 403,
		},
		{
			name: "violation: excluding matchConditions are not evaluated", reqs: dynamic,
			rules: []v1.RuleWithOperations{podRule(createUpdate, "pods")}, matchConditions: matchConditions, matchResult: false,
			attrs: target(), expectCode: 403, expectMsg: []string{"intercepts CREATE pods"},
		},
		{
			name: "violation: erroring matchConditions are not evaluated", reqs: dynamic,
			rules: []v1.RuleWithOperations{podRule(createUpdate, "pods")}, matchConditions: matchConditions, matchError: errors.New("test error"),
			attrs: target(), expectCode: 403, expectMsg: []string{"intercepts CREATE pods"}, rejectMsg: []string{"test error"},
		},
		{
			name: "violation: versioned attributes from the mutating dispatcher", reqs: dynamic,
			rules: []v1.RuleWithOperations{podRule(createUpdate, "pods")}, attrs: target(), versioned: true,
			expectCode: 403, expectMsg: []string{"intercepts CREATE pods but not UPDATE pods/dynamic"},
			expectAnnotation: `[{"name":"hook.example.com","configuration":"cfg","source":"CREATE pods","target":"UPDATE pods/dynamic"}]`,
		},
		{
			name: "main-resource target: hint has no trailing slash", reqs: mainResource,
			rules:      []v1.RuleWithOperations{podRule([]v1.OperationType{v1.Create}, "pods")},
			attrs:      webhooktesting.NewPodAttribute(admission.Update, "", nil),
			expectCode: 403, expectMsg: []string{"intercepts CREATE pods but not UPDATE pods", `add a rule for UPDATE on "pods" (or "pods/*") to the webhook's rules`},
			rejectMsg: []string{`"pods/"`},
		},

		// Direct matches are called normally.
		{name: "direct match: pods/*", reqs: dynamic, rules: []v1.RuleWithOperations{podRule(createUpdate, "pods/*")}, attrs: target(), matchResult: true, expectCall: true},
		{name: "direct match: */*", reqs: dynamic, rules: []v1.RuleWithOperations{podRule(createUpdate, "*/*")}, attrs: target(), matchResult: true, expectCall: true},
		{name: "direct match: explicit target", reqs: dynamic, rules: []v1.RuleWithOperations{podRule(createUpdate, "pods", "pods/dynamic")}, attrs: target(), matchResult: true, expectCall: true},
		{
			name: "direct match: matchConditions still apply", reqs: dynamic,
			rules: []v1.RuleWithOperations{podRule(createUpdate, "pods", "pods/dynamic")}, matchConditions: matchConditions, matchResult: false,
			attrs: target(),
		},

		// Directionality.
		{name: "target-only hook, UPDATE pods source request: not called", reqs: dynamic, rules: []v1.RuleWithOperations{podRule(update, "pods/dynamic")}, attrs: webhooktesting.NewPodAttribute(admission.Update, "", nil)},
		{name: "violating hook, source request: called normally", reqs: dynamic, rules: []v1.RuleWithOperations{podRule(createUpdate, "pods")}, attrs: webhooktesting.NewPodAttribute(admission.Update, "", nil), matchResult: true, expectCall: true},

		// matchPolicy. The gaps ignore it, so they flag the target either way;
		// only a hook that matches the real request through Equivalent is called.
		{
			name: "matchPolicy Equivalent: hook with gaps matching the target via another group is called", reqs: deploymentScale,
			rules: equivalentScaleRules, matchPolicy: &equivalent, mapper: equivalentMapper,
			attrs: deploymentScaleTarget(), matchResult: true, expectCall: true,
		},
		{
			name: "matchPolicy Exact: same hook is in violation", reqs: deploymentScale,
			rules: equivalentScaleRules, matchPolicy: &exact, mapper: equivalentMapper,
			attrs: deploymentScaleTarget(), expectCode: 403,
			expectMsg: []string{`intercepts CREATE deployments.apps but not UPDATE deployments.apps/scale; add a rule for UPDATE on "deployments/scale" (or "deployments/*")`},
		},

		// Selectors.
		{
			name: "namespace selector excludes: no violation", reqs: dynamic,
			rules:             []v1.RuleWithOperations{podRule(createUpdate, "pods")},
			namespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"env": "dev"}}, attrs: target(),
		},
		{
			name: "object selector excludes: no violation", reqs: dynamic,
			rules:          []v1.RuleWithOperations{podRule(createUpdate, "pods")},
			objectSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "x"}}, attrs: target(),
		},
		{
			name: "deferred namespace selector error preferred over coverage", reqs: dynamic,
			rules: []v1.RuleWithOperations{podRule(createUpdate, "pods")}, namespaceSelector: badSelector, attrs: target(),
			expectCode: 500, rejectMsg: []string{"intercepts"},
		},
		{
			name: "deferred object selector error preferred over coverage", reqs: dynamic,
			rules: []v1.RuleWithOperations{podRule(createUpdate, "pods")}, objectSelector: badSelector, attrs: target(),
			expectCode: 500, rejectMsg: []string{"intercepts"},
		},
		{
			name: "selector error on a hook not in violation stays hidden", reqs: dynamic,
			rules: []v1.RuleWithOperations{podRule(update, "pods/resize")}, namespaceSelector: badSelector, attrs: target(),
		},

		// Exemptions and no requirements.
		{name: "mutating reinvocation is exempt", reqs: dynamic, rules: []v1.RuleWithOperations{podRule(createUpdate, "pods")}, attrs: target(), reinvoke: true},
		{name: "no requirements: violating hook not called, no error", rules: []v1.RuleWithOperations{podRule(createUpdate, "pods")}, attrs: target()},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			a := &Webhook{
				namespaceMatcher: &namespace.Matcher{NamespaceLister: namespaceLister},
				objectMatcher:    &object.Matcher{},
			}
			a.SetAdmissionCoverage(tc.reqs)

			nsSel, objSel := tc.namespaceSelector, tc.objectSelector
			if nsSel == nil {
				nsSel = &metav1.LabelSelector{}
			}
			if objSel == nil {
				objSel = &metav1.LabelSelector{}
			}
			h := &fakeWebhookAccessor{
				WebhookAccessor: webhook.NewValidatingWebhookAccessor("cfg/hook.example.com/0", "cfg", &v1.ValidatingWebhook{
					Name: "hook.example.com", Rules: tc.rules, MatchPolicy: tc.matchPolicy,
					NamespaceSelector: nsSel, ObjectSelector: objSel, MatchConditions: tc.matchConditions,
				}),
				matchResult: tc.matchResult,
				throwError:  tc.matchError,
			}
			if tc.reinvoke {
				tc.attrs.GetReinvocationContext().SetIsReinvoke()
			}
			var attrs admission.Attributes = tc.attrs
			if tc.versioned {
				v := &admission.VersionedAttributes{Attributes: tc.attrs, VersionedKind: tc.attrs.GetKind()}
				v.VersionedObject.Set(tc.attrs.GetObject().DeepCopyObject())
				v.VersionedOldObject.Set(tc.attrs.GetOldObject().DeepCopyObject())
				attrs = v
			}

			objectInterfaces := interfaces
			if tc.mapper != nil {
				objectInterfaces = &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: tc.mapper}
			}
			invocation, err := a.ShouldCallHook(t.Context(), h, attrs, objectInterfaces, &fakeVersionedAttributeAccessor{})
			if tc.expectCode == 0 {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Fatalf("expected %d error, got invocation %v", tc.expectCode, invocation)
				}
				if err.ErrStatus.Code != tc.expectCode {
					t.Fatalf("expected code %d, got %d: %v", tc.expectCode, err.ErrStatus.Code, err)
				}
			}
			for _, s := range tc.expectMsg {
				if err == nil || !strings.Contains(err.Error(), s) {
					t.Errorf("expected error containing %q, got %v", s, err)
				}
			}
			for _, s := range tc.rejectMsg {
				if err != nil && strings.Contains(err.Error(), s) {
					t.Errorf("expected error not containing %q, got %v", s, err)
				}
			}
			if (invocation != nil) != tc.expectCall {
				t.Errorf("expected call=%v, got invocation %v", tc.expectCall, invocation)
			}

			annotation, annotated := tc.attrs.GetAnnotations(auditinternal.LevelMetadata)[annotationKey]
			if want := tc.expectCode == 403; annotated != want {
				t.Errorf("expected coverage audit annotation: %v, got %q (present=%v)", want, annotation, annotated)
			}
			if tc.expectAnnotation != "" && annotation != tc.expectAnnotation {
				t.Errorf("expected coverage audit annotation %s, got %s", tc.expectAnnotation, annotation)
			}
		})
	}
}

func BenchmarkShouldCallHookWithComplexSelector(b *testing.B) {
	allScopes := v1.AllScopes
	equivalentMatch := v1.Equivalent

	namespace1Labels := map[string]string{"ns": "ns1"}
	namespace1 := corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "ns1",
			Labels: namespace1Labels,
		},
	}
	namespaceLister := fakeNamespaceLister{map[string]*corev1.Namespace{"ns": &namespace1}}

	mapper := runtime.NewEquivalentResourceRegistryWithIdentity(func(resource schema.GroupResource) string {
		if resource.Resource == "deployments" {
			// co-locate deployments in all API groups
			return "/deployments"
		}
		return ""
	})
	mapper.RegisterKindFor(gvr("extensions", "v1beta1", "deployments"), "", gvk("extensions", "v1beta1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1", "deployments"), "", gvk("apps", "v1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "deployments"), "", gvk("apps", "v1beta1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha1", "deployments"), "", gvk("apps", "v1alpha1", "Deployment"))

	mapper.RegisterKindFor(gvr("extensions", "v1beta1", "deployments"), "scale", gvk("extensions", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1", "deployments"), "scale", gvk("autoscaling", "v1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "deployments"), "scale", gvk("apps", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha1", "deployments"), "scale", gvk("apps", "v1alpha1", "Scale"))

	mapper.RegisterKindFor(gvr("apps", "v1", "statefulset"), "", gvk("apps", "v1", "StatefulSet"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "statefulset"), "", gvk("apps", "v1beta1", "StatefulSet"))
	mapper.RegisterKindFor(gvr("apps", "v1beta2", "statefulset"), "", gvk("apps", "v1beta2", "StatefulSet"))

	mapper.RegisterKindFor(gvr("apps", "v1", "statefulset"), "scale", gvk("apps", "v1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "statefulset"), "scale", gvk("apps", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha2", "statefulset"), "scale", gvk("apps", "v1beta2", "Scale"))

	nsSelector := make(map[string]string)
	for i := 0; i < 100; i++ {
		nsSelector[fmt.Sprintf("key-%d", i)] = fmt.Sprintf("val-%d", i)
	}

	wb := &v1.ValidatingWebhook{
		MatchPolicy:       &equivalentMatch,
		NamespaceSelector: &metav1.LabelSelector{MatchLabels: nsSelector},
		ObjectSelector:    &metav1.LabelSelector{},
		Rules: []v1.RuleWithOperations{
			{
				Operations: []v1.OperationType{"*"},
				Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
			},
			{
				Operations: []v1.OperationType{"*"},
				Rule:       v1.Rule{APIGroups: []string{"extensions"}, APIVersions: []string{"v1beta1"}, Resources: []string{"deployments", "deployments/scale"}, Scope: &allScopes},
			},
		},
	}

	wbAccessor := &fakeWebhookAccessor{
		WebhookAccessor: webhook.NewValidatingWebhookAccessor("webhook", "webhook-cfg", wb),
		matchResult:     true,
	}
	attrs := admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "ns", "name", gvr("apps", "v1", "deployments"), "scale", admission.Create, &metav1.CreateOptions{}, false, nil)
	interfaces := &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: mapper}
	a := &Webhook{namespaceMatcher: &namespace.Matcher{NamespaceLister: namespaceLister}, objectMatcher: &object.Matcher{}}

	for i := 0; i < b.N; i++ {
		a.ShouldCallHook(context.TODO(), wbAccessor, attrs, interfaces, nil)
	}
}

func BenchmarkShouldCallHookWithComplexRule(b *testing.B) {
	allScopes := v1.AllScopes
	equivalentMatch := v1.Equivalent

	namespace1Labels := map[string]string{"ns": "ns1"}
	namespace1 := corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "ns1",
			Labels: namespace1Labels,
		},
	}
	namespaceLister := fakeNamespaceLister{map[string]*corev1.Namespace{"ns": &namespace1}}

	mapper := runtime.NewEquivalentResourceRegistryWithIdentity(func(resource schema.GroupResource) string {
		if resource.Resource == "deployments" {
			// co-locate deployments in all API groups
			return "/deployments"
		}
		return ""
	})
	mapper.RegisterKindFor(gvr("extensions", "v1beta1", "deployments"), "", gvk("extensions", "v1beta1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1", "deployments"), "", gvk("apps", "v1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "deployments"), "", gvk("apps", "v1beta1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha1", "deployments"), "", gvk("apps", "v1alpha1", "Deployment"))

	mapper.RegisterKindFor(gvr("extensions", "v1beta1", "deployments"), "scale", gvk("extensions", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1", "deployments"), "scale", gvk("autoscaling", "v1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "deployments"), "scale", gvk("apps", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha1", "deployments"), "scale", gvk("apps", "v1alpha1", "Scale"))

	mapper.RegisterKindFor(gvr("apps", "v1", "statefulset"), "", gvk("apps", "v1", "StatefulSet"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "statefulset"), "", gvk("apps", "v1beta1", "StatefulSet"))
	mapper.RegisterKindFor(gvr("apps", "v1beta2", "statefulset"), "", gvk("apps", "v1beta2", "StatefulSet"))

	mapper.RegisterKindFor(gvr("apps", "v1", "statefulset"), "scale", gvk("apps", "v1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "statefulset"), "scale", gvk("apps", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha2", "statefulset"), "scale", gvk("apps", "v1beta2", "Scale"))

	wb := &v1.ValidatingWebhook{
		MatchPolicy:       &equivalentMatch,
		NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
		ObjectSelector:    &metav1.LabelSelector{},
		Rules:             []v1.RuleWithOperations{},
	}

	for i := 0; i < 100; i++ {
		rule := v1.RuleWithOperations{
			Operations: []v1.OperationType{"*"},
			Rule: v1.Rule{
				APIGroups:   []string{fmt.Sprintf("app-%d", i)},
				APIVersions: []string{fmt.Sprintf("v%d", i)},
				Resources:   []string{fmt.Sprintf("resource%d", i), fmt.Sprintf("resource%d/scale", i)},
				Scope:       &allScopes,
			},
		}
		wb.Rules = append(wb.Rules, rule)
	}

	wbAccessor := &fakeWebhookAccessor{
		WebhookAccessor: webhook.NewValidatingWebhookAccessor("webhook", "webhook-cfg", wb),
		matchResult:     true,
	}
	attrs := admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "ns", "name", gvr("apps", "v1", "deployments"), "scale", admission.Create, &metav1.CreateOptions{}, false, nil)
	interfaces := &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: mapper}
	a := &Webhook{namespaceMatcher: &namespace.Matcher{NamespaceLister: namespaceLister}, objectMatcher: &object.Matcher{}}

	for i := 0; i < b.N; i++ {
		a.ShouldCallHook(context.TODO(), wbAccessor, attrs, interfaces, &fakeVersionedAttributeAccessor{})
	}
}

func BenchmarkShouldCallHookWithComplexSelectorAndRule(b *testing.B) {
	allScopes := v1.AllScopes
	equivalentMatch := v1.Equivalent

	namespace1Labels := map[string]string{"ns": "ns1"}
	namespace1 := corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "ns1",
			Labels: namespace1Labels,
		},
	}
	namespaceLister := fakeNamespaceLister{map[string]*corev1.Namespace{"ns": &namespace1}}

	mapper := runtime.NewEquivalentResourceRegistryWithIdentity(func(resource schema.GroupResource) string {
		if resource.Resource == "deployments" {
			// co-locate deployments in all API groups
			return "/deployments"
		}
		return ""
	})
	mapper.RegisterKindFor(gvr("extensions", "v1beta1", "deployments"), "", gvk("extensions", "v1beta1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1", "deployments"), "", gvk("apps", "v1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "deployments"), "", gvk("apps", "v1beta1", "Deployment"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha1", "deployments"), "", gvk("apps", "v1alpha1", "Deployment"))

	mapper.RegisterKindFor(gvr("extensions", "v1beta1", "deployments"), "scale", gvk("extensions", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1", "deployments"), "scale", gvk("autoscaling", "v1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "deployments"), "scale", gvk("apps", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha1", "deployments"), "scale", gvk("apps", "v1alpha1", "Scale"))

	mapper.RegisterKindFor(gvr("apps", "v1", "statefulset"), "", gvk("apps", "v1", "StatefulSet"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "statefulset"), "", gvk("apps", "v1beta1", "StatefulSet"))
	mapper.RegisterKindFor(gvr("apps", "v1beta2", "statefulset"), "", gvk("apps", "v1beta2", "StatefulSet"))

	mapper.RegisterKindFor(gvr("apps", "v1", "statefulset"), "scale", gvk("apps", "v1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1beta1", "statefulset"), "scale", gvk("apps", "v1beta1", "Scale"))
	mapper.RegisterKindFor(gvr("apps", "v1alpha2", "statefulset"), "scale", gvk("apps", "v1beta2", "Scale"))

	nsSelector := make(map[string]string)
	for i := 0; i < 100; i++ {
		nsSelector[fmt.Sprintf("key-%d", i)] = fmt.Sprintf("val-%d", i)
	}

	wb := &v1.ValidatingWebhook{
		MatchPolicy:       &equivalentMatch,
		NamespaceSelector: &metav1.LabelSelector{MatchLabels: nsSelector},
		ObjectSelector:    &metav1.LabelSelector{},
		Rules:             []v1.RuleWithOperations{},
	}

	for i := 0; i < 100; i++ {
		rule := v1.RuleWithOperations{
			Operations: []v1.OperationType{"*"},
			Rule: v1.Rule{
				APIGroups:   []string{fmt.Sprintf("app-%d", i)},
				APIVersions: []string{fmt.Sprintf("v%d", i)},
				Resources:   []string{fmt.Sprintf("resource%d", i), fmt.Sprintf("resource%d/scale", i)},
				Scope:       &allScopes,
			},
		}
		wb.Rules = append(wb.Rules, rule)
	}

	wbAccessor := &fakeWebhookAccessor{
		WebhookAccessor: webhook.NewValidatingWebhookAccessor("webhook", "webhook-cfg", wb),
		matchResult:     true,
	}
	attrs := admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "ns", "name", gvr("apps", "v1", "deployments"), "scale", admission.Create, &metav1.CreateOptions{}, false, nil)
	interfaces := &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: mapper}
	a := &Webhook{namespaceMatcher: &namespace.Matcher{NamespaceLister: namespaceLister}, objectMatcher: &object.Matcher{}}

	for i := 0; i < b.N; i++ {
		a.ShouldCallHook(context.TODO(), wbAccessor, attrs, interfaces, nil)
	}
}

// BenchmarkShouldCallHookCoverage measures the coverage check for a hook whose
// rules don't match the request. Its selectors are empty, unlike the
// benchmarks above, so that ShouldCallHook gets past them to the check.
func BenchmarkShouldCallHookCoverage(b *testing.B) {
	allScopes := v1.AllScopes
	wb := &v1.ValidatingWebhook{
		NamespaceSelector: &metav1.LabelSelector{},
		ObjectSelector:    &metav1.LabelSelector{},
		Rules: []v1.RuleWithOperations{{
			Operations: []v1.OperationType{v1.Create, v1.Update},
			Rule:       v1.Rule{APIGroups: []string{"apps"}, APIVersions: []string{"*"}, Resources: []string{"deployments"}, Scope: &allScopes},
		}},
	}
	reqs := []coverage.Requirement{{
		Resource: schema.GroupResource{Group: "apps", Resource: "deployments"},
		Target:   coverage.Shape{Operation: admission.Update, Subresource: "scale"},
		Sources:  []coverage.Shape{{Operation: admission.Create}, {Operation: admission.Update}},
	}}
	newAttrs := func(subresource string) admission.Attributes {
		return admission.NewAttributesRecord(nil, nil, gvk("autoscaling", "v1", "Scale"), "ns", "name", gvr("apps", "v1", "deployments"), subresource, admission.Update, &metav1.UpdateOptions{}, false, nil)
	}
	reinvocation := newAttrs("scale")
	reinvocation.GetReinvocationContext().SetIsReinvoke()
	interfaces := &admission.RuntimeObjectInterfaces{EquivalentResourceMapper: runtime.NewEquivalentResourceRegistry()}

	for _, tc := range []struct {
		name  string
		reqs  []coverage.Requirement
		attrs admission.Attributes
	}{
		{name: "no requirements", attrs: newAttrs("status")},
		{name: "non-target", reqs: reqs, attrs: newAttrs("status")},
		{name: "reinvocation", reqs: reqs, attrs: reinvocation},
		{name: "target violation", reqs: reqs, attrs: newAttrs("scale")},
	} {
		b.Run(tc.name, func(b *testing.B) {
			// A fresh accessor per sub-benchmark: GetCoverageGaps keeps the gaps of the first requirements.
			wbAccessor := &fakeWebhookAccessor{WebhookAccessor: webhook.NewValidatingWebhookAccessor("webhook", "webhook-cfg", wb)}
			a := &Webhook{namespaceMatcher: &namespace.Matcher{}, objectMatcher: &object.Matcher{}}
			a.SetAdmissionCoverage(tc.reqs)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _ = a.ShouldCallHook(b.Context(), wbAccessor, tc.attrs, interfaces, nil)
			}
		})
	}
}
