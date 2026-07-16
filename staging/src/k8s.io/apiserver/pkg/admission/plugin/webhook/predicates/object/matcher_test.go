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

package object_test

import (
	"testing"

	"k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/object"
)

func TestObjectSelector(t *testing.T) {
	nodeLevel1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				"runlevel": "1",
			},
		},
	}
	nodeLevel2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				"runlevel": "2",
			},
		},
	}
	runLevel1Excluder := &metav1.LabelSelector{
		MatchExpressions: []metav1.LabelSelectorRequirement{
			{
				Key:      "runlevel",
				Operator: metav1.LabelSelectorOpNotIn,
				Values:   []string{"1"},
			},
		},
	}
	matcher := &object.Matcher{}
	allScopes := v1.AllScopes
	testcases := []struct {
		name string

		objectSelector *metav1.LabelSelector
		attrs          admission.Attributes

		expectCall bool
	}{
		{
			name:           "empty object selector matches everything",
			objectSelector: &metav1.LabelSelector{},
			attrs:          admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, "", "name", schema.GroupVersionResource{}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:     true,
		},
		{
			name:           "matches new object",
			objectSelector: runLevel1Excluder,
			attrs:          admission.NewAttributesRecord(nodeLevel2, nil, schema.GroupVersionKind{}, "", "name", schema.GroupVersionResource{}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:     true,
		},
		{
			name:           "matches old object",
			objectSelector: runLevel1Excluder,
			attrs:          admission.NewAttributesRecord(nil, nodeLevel2, schema.GroupVersionKind{}, "", "name", schema.GroupVersionResource{}, "", admission.Delete, &metav1.DeleteOptions{}, false, nil),
			expectCall:     true,
		},
		{
			name:           "does not match new object",
			objectSelector: runLevel1Excluder,
			attrs:          admission.NewAttributesRecord(nodeLevel1, nil, schema.GroupVersionKind{}, "", "name", schema.GroupVersionResource{}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:     false,
		},
		{
			name:           "does not match old object",
			objectSelector: runLevel1Excluder,
			attrs:          admission.NewAttributesRecord(nil, nodeLevel1, schema.GroupVersionKind{}, "", "name", schema.GroupVersionResource{}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:     false,
		},
		{
			name:           "does not match object that does not implement Object interface",
			objectSelector: runLevel1Excluder,
			attrs:          admission.NewAttributesRecord(&corev1.NodeProxyOptions{}, nil, schema.GroupVersionKind{}, "", "name", schema.GroupVersionResource{}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:     false,
		},
		{
			name:           "empty selector matches everything, including object that does not implement Object interface",
			objectSelector: &metav1.LabelSelector{},
			attrs:          admission.NewAttributesRecord(&corev1.NodeProxyOptions{}, nil, schema.GroupVersionKind{}, "", "name", schema.GroupVersionResource{}, "", admission.Create, &metav1.CreateOptions{}, false, nil),
			expectCall:     true,
		},
	}

	for _, testcase := range testcases {
		hook := &v1.ValidatingWebhook{
			NamespaceSelector: &metav1.LabelSelector{},
			ObjectSelector:    testcase.objectSelector,
			Rules: []v1.RuleWithOperations{{
				Operations: []v1.OperationType{"*"},
				Rule:       v1.Rule{APIGroups: []string{"*"}, APIVersions: []string{"*"}, Resources: []string{"*"}, Scope: &allScopes},
			}}}

		t.Run(testcase.name, func(t *testing.T) {
			match, err := matcher.MatchObjectSelector(webhook.NewValidatingWebhookAccessor("mock-hook", "mock-cfg", hook), testcase.attrs)
			if err != nil {
				t.Error(err)
			}
			if testcase.expectCall && !match {
				t.Errorf("expected the webhook to be called")
			}
			if !testcase.expectCall && match {
				t.Errorf("expected the webhook to be called")
			}
		})
	}
}
