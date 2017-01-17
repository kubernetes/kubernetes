/*
Copyright 2017 The Kubernetes Authors.

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

package set

import (
	"bytes"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/rbac"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestValidate(t *testing.T) {
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Namespace = "test"

	tests := map[string]struct {
		ruleOptions *RuleOptions
		expectErr   bool
	}{
		"test-missing-verb": {
			ruleOptions: &RuleOptions{},
			expectErr:   true,
		},
		"test-missing-resource": {
			ruleOptions: &RuleOptions{
				Verbs: []string{"get"},
			},
			expectErr: true,
		},
		"test-invalid-verb-for-resource": {
			ruleOptions: &RuleOptions{
				Verbs: []string{"invalid-verb"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
					},
				},
			},
			expectErr: true,
		},
		"test-invalid-for-nonresource": {
			ruleOptions: &RuleOptions{
				Verbs:           []string{"list"},
				NonResourceURLs: []string{"test"},
			},
			expectErr: true,
		},
		"test-invalid-resource": {
			ruleOptions: &RuleOptions{
				Verbs: []string{"get"},
				Resources: []ResourceOptions{
					{
						Resource: "invalid-resource",
					},
				},
			},
			expectErr: true,
		},
		"test-invalid-subresource": {
			ruleOptions: &RuleOptions{
				Verbs: []string{"get"},
				Resources: []ResourceOptions{
					{
						Resource:    "replicasets",
						SubResource: "invalid-resource",
					},
				},
			},
			expectErr: true,
		},
		"test-resource-name-with-multiple-Resources": {
			ruleOptions: &RuleOptions{
				Verbs: []string{"get"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				ResourceNames: []string{"foo"},
			},
			expectErr: true,
		},
		"test-valid-case": {
			ruleOptions: &RuleOptions{
				Verbs: []string{"get", "list"},
				Resources: []ResourceOptions{
					{
						Resource:    "replicasets",
						SubResource: "scales",
					},
				},
				ResourceNames: []string{"foo"},
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		err := test.ruleOptions.Validate(f)
		if test.expectErr && err != nil {
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
	}
}

func TestNewPolicyRuleForObject(t *testing.T) {
	tests := []struct {
		name     string
		obj      runtime.Object
		rule     *rbac.PolicyRule
		expected []rbac.PolicyRule
		wantErr  bool
	}{
		{
			name: "empty resource in role",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods"},
					Verbs:     []string{"get"},
				},
			},
			wantErr: false,
		},
		{
			name: "empty resource in clusterrole",
			obj: &rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: "clusterrole",
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods"},
					Verbs:     []string{"get"},
				},
			},
			wantErr: false,
		},
		{
			name: "empty non resource in clusterrole",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name: "clusterrole",
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/*"},
					Verbs:           []string{"get"},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		if err := updatePolicyRuleForObject(tt.obj, tt.rule, addPolicyRule); (err != nil) != tt.wantErr {
			t.Errorf("%q. updatePolicyRuleForObject() error = %v, wantErr %v", tt.name, err, tt.wantErr)
		}

		want := tt.expected
		var got []rbac.PolicyRule
		switch t := tt.obj.(type) {
		case *rbac.Role:
			got = t.Rules
		case *rbac.ClusterRole:
			got = t.Rules
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q. updatePolicyRuleForObject() failed", tt.name)
			t.Errorf("Got: %v", got)
			t.Errorf("Want: %v", want)
		}
	}
}

func TestUpdatePolicyRuleForObject(t *testing.T) {
	tests := []struct {
		name     string
		obj      runtime.Object
		rule     *rbac.PolicyRule
		expected []rbac.PolicyRule
		wantErr  bool
	}{
		{
			name: "update resource in role",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{""},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{""},
					Resources: []string{"pods"},
					Verbs:     []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
		{
			name: "add resource in clusterrole",
			obj: &rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: "clusterrole",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{""},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{""},
					Resources: []string{"pods"},
					Verbs:     []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
		{
			name: "add non resource in clusterrole",
			obj: &rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: "clusterrole",
				},
				Rules: []rbac.PolicyRule{
					{
						NonResourceURLs: []string{"/apis/*", "/apis/v1"},
						Verbs:           []string{"delete"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/v1"},
					Verbs:           []string{"delete"},
				},
				{
					NonResourceURLs: []string{"/apis/*"},
					Verbs:           []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		if err := updatePolicyRuleForObject(tt.obj, tt.rule, addPolicyRule); (err != nil) != tt.wantErr {
			t.Errorf("%q. updatePolicyRuleForObject() error = %v, wantErr %v", tt.name, err, tt.wantErr)
		}

		want := tt.expected
		var got []rbac.PolicyRule
		switch t := tt.obj.(type) {
		case *rbac.Role:
			got = t.Rules
		case *rbac.ClusterRole:
			got = t.Rules
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q. updatePolicyRuleForObject() failed", tt.name)
			t.Errorf("Got: %v", got)
			t.Errorf("Want: %v", want)
		}
	}
}

func TestAddPolicyRule(t *testing.T) {
	tests := []struct {
		name     string
		existing []rbac.PolicyRule
		rule     *rbac.PolicyRule
		expected []rbac.PolicyRule
		wantErr  bool
	}{
		{
			name: "add resource with union Resources in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{""},
					Resources: []string{"pods"},
					Verbs:     []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
		{
			name: "add resource without union Resources in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"apps"},
				Resources: []string{"deployments"},
				Verbs:     []string{"delete"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{"apps"},
					Resources: []string{"deployments"},
					Verbs:     []string{"delete"},
				},
			},
			wantErr: false,
		},
		{
			name: "add resource with same resource name in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods", "namespaces"},
					Verbs:         []string{"delete", "get"},
					ResourceNames: []string{"foo"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups:     []string{""},
				Resources:     []string{"pods", "namespaces"},
				Verbs:         []string{"list"},
				ResourceNames: []string{"foo"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods", "namespaces"},
					Verbs:         []string{"delete", "get", "list"},
					ResourceNames: []string{"foo"},
				},
			},
			wantErr: false,
		},
		{
			name: "add resource with different resource name in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods", "namespaces"},
					Verbs:         []string{"delete", "get"},
					ResourceNames: []string{"foo"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups:     []string{""},
				Resources:     []string{"pods", "namespaces"},
				Verbs:         []string{"list"},
				ResourceNames: []string{"bar"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods", "namespaces"},
					Verbs:         []string{"delete", "get"},
					ResourceNames: []string{"foo"},
				},
				{
					APIGroups:     []string{""},
					Resources:     []string{"pods", "namespaces"},
					Verbs:         []string{"list"},
					ResourceNames: []string{"bar"},
				},
			},
			wantErr: false,
		},
		{
			name: "add resource with wildcard permission in role",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"*"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{""},
					Resources: []string{"pods"},
					Verbs:     []string{"*"},
				},
			},
			wantErr: false,
		},
		{
			name: "add resource in clusterrole",
			existing: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{""},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{""},
					Resources: []string{"pods"},
					Verbs:     []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
		{
			name: "add non resource in clusterrole",
			existing: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/*", "/apis/v1"},
					Verbs:           []string{"delete"},
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/v1"},
					Verbs:           []string{"delete"},
				},
				{
					NonResourceURLs: []string{"/apis/*"},
					Verbs:           []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
		{
			name: "add non resource with wildcard permission in clusterrole",
			existing: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/*", "/apis/v1"},
					Verbs:           []string{"delete"},
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"*"},
			},
			expected: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/v1"},
					Verbs:           []string{"delete"},
				},
				{
					NonResourceURLs: []string{"/apis/*"},
					Verbs:           []string{"*"},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		var err error
		got := []rbac.PolicyRule{}
		if got, err = addPolicyRule(tt.existing, tt.rule); (err != nil) != tt.wantErr {
			t.Errorf("%q. addPolicyRule() error = %v, wantErr %v", tt.name, err, tt.wantErr)
		}
		want := tt.expected
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q. addPolicyRule() failed", tt.name)
			t.Errorf("Got: %v", got)
			t.Errorf("Want: %v", want)
		}
	}
}

func TestUpdateRuleOptionsWithResource(t *testing.T) {
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Namespace = "test"

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdRule(f, buf, buf)
	cmd.Flags().Set("resource", "pods,deployments.extensions/scale")

	tests := map[string]struct {
		ruleOptions *RuleOptions
		expected    *RuleOptions
		expectErr   bool
	}{
		"test-duplicate-Verbs": {
			ruleOptions: &RuleOptions{
				Verbs: []string{
					"get",
					"watch",
					"list",
					"get",
				},
			},
			expected: &RuleOptions{
				Verbs: []string{
					"get",
					"watch",
					"list",
				},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource:    "deployments",
						Group:       "extensions",
						SubResource: "scale",
					},
				},
				ResourceNames:   []string{},
				NonResourceURLs: []string{},
			},
			expectErr: false,
		},
		"test-verball": {
			ruleOptions: &RuleOptions{
				Verbs: []string{
					"get",
					"watch",
					"list",
					"*",
				},
			},
			expected: &RuleOptions{
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource:    "deployments",
						Group:       "extensions",
						SubResource: "scale",
					},
				},
				ResourceNames:   []string{},
				NonResourceURLs: []string{},
			},
			expectErr: false,
		},
		"test-duplicate-resourcenames": {
			ruleOptions: &RuleOptions{
				Verbs:         []string{"*"},
				ResourceNames: []string{"foo", "foo"},
			},
			expected: &RuleOptions{
				Verbs: []string{"*"},
				Resources: []ResourceOptions{
					{
						Resource: "pods",
						Group:    "",
					},
					{
						Resource:    "deployments",
						Group:       "extensions",
						SubResource: "scale",
					},
				},
				ResourceNames:   []string{"foo"},
				NonResourceURLs: []string{},
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		err := test.ruleOptions.UpdateRuleOptions(cmd)
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(test.ruleOptions, test.expected) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expected, test.ruleOptions)
		}
	}
}

func TestUpdateRuleOptionsWithNonResource(t *testing.T) {
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Namespace = "test"

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdRule(f, buf, buf)

	tests := map[string]struct {
		ruleOptions *RuleOptions
		expected    *RuleOptions
		expectErr   bool
	}{
		"test-duplicate-Verbs": {
			ruleOptions: &RuleOptions{
				Verbs: []string{
					"get",
					"put",
					"post",
					"get",
				},
				NonResourceURLs: []string{"test", "test"},
			},
			expected: &RuleOptions{
				Verbs: []string{
					"get",
					"put",
					"post",
				},
				Resources:       []ResourceOptions{},
				ResourceNames:   []string{},
				NonResourceURLs: []string{"test"},
			},
			expectErr: false,
		},
		"test-verball": {
			ruleOptions: &RuleOptions{
				Verbs: []string{
					"get",
					"put",
					"post",
					"*",
				},
				NonResourceURLs: []string{"test"},
			},
			expected: &RuleOptions{
				Verbs:           []string{"*"},
				Resources:       []ResourceOptions{},
				ResourceNames:   []string{},
				NonResourceURLs: []string{"test"},
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		err := test.ruleOptions.UpdateRuleOptions(cmd)
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(test.ruleOptions, test.expected) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", name, test.expected, test.ruleOptions)
		}
	}
}
