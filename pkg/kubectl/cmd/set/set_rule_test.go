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
				APIGroups: []string{"extensions"},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"extensions"},
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
				APIGroups: []string{"extensions"},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"extensions"},
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
		if err := updatePolicyRuleForObject(tt.obj, tt.rule, false); (err != nil) != tt.wantErr {
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

func TestAddPolicyRuleForObject(t *testing.T) {
	tests := []struct {
		name     string
		obj      runtime.Object
		rule     *rbac.PolicyRule
		expected []rbac.PolicyRule
		wantErr  bool
	}{
		{
			name: "add resource with union resources in role",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"extensions"},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"extensions"},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{"extensions"},
					Resources: []string{"pods"},
					Verbs:     []string{"delete", "get"},
				},
				{
					APIGroups: []string{"apps"},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
				},
			},
			wantErr: false,
		},
		{
			name: "add resource without union resources in role",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{""},
				Resources: []string{"nodes"},
				Verbs:     []string{"delete"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"extensions", "apps"},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{""},
					Resources: []string{"nodes"},
					Verbs:     []string{"delete"},
				},
			},
			wantErr: false,
		},
		{
			name: "add resource with wildcard permission in role",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"extensions"},
				Resources: []string{"pods"},
				Verbs:     []string{"*"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"extensions"},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{"extensions"},
					Resources: []string{"pods"},
					Verbs:     []string{"*"},
				},
				{
					APIGroups: []string{"apps"},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
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
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"extensions"},
				Resources: []string{"pods"},
				Verbs:     []string{"get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"extensions"},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{"extensions"},
					Resources: []string{"pods"},
					Verbs:     []string{"delete", "get"},
				},
				{
					APIGroups: []string{"apps"},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
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
		{
			name: "add non resource with wildcard permission in clusterrole",
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
		if err := updatePolicyRuleForObject(tt.obj, tt.rule, false); (err != nil) != tt.wantErr {
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

func TestRemovePolicyRuleForObject(t *testing.T) {
	tests := []struct {
		name     string
		obj      runtime.Object
		rule     *rbac.PolicyRule
		expected []rbac.PolicyRule
		wantErr  bool
	}{
		{
			name: "remove resource with same verbs in role",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"extensions"},
				Resources: []string{"pods"},
				Verbs:     []string{"delete"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"extensions"},
					Resources: []string{"namespaces"},
					Verbs:     []string{"delete"},
				},
				{
					APIGroups: []string{"apps"},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove resource with same resources in role",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete", "get"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"extensions"},
				Resources: []string{"pods", "namespaces"},
				Verbs:     []string{"delete"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"extensions"},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"get"},
				},
				{
					APIGroups: []string{"apps"},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove resource with wildcard permission in role",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"*"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"extensions"},
				Resources: []string{"pods"},
				Verbs:     []string{"create", "get", "delete", "list", "watch"},
			},
			wantErr: true,
		},
		{
			name: "remove wildcard permission resource from role",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete", "get"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"extensions"},
				Resources: []string{"pods"},
				Verbs:     []string{"*"},
			},
			wantErr: true,
		},
		{
			name: "remove wildcard permission resource from role which have wildcard",
			obj: &rbac.Role{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "role",
					Namespace: "one",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"*"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"extensions"},
				Resources: []string{"pods"},
				Verbs:     []string{"*"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"extensions"},
					Resources: []string{"namespaces"},
					Verbs:     []string{"*"},
				},
				{
					APIGroups: []string{"apps"},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"*"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove resource in clusterrole",
			obj: &rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: "clusterrole",
				},
				Rules: []rbac.PolicyRule{
					{
						APIGroups: []string{"extensions", "apps"},
						Resources: []string{"pods", "namespaces"},
						Verbs:     []string{"delete", "get"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				APIGroups: []string{"extensions"},
				Resources: []string{"pods", "namespaces"},
				Verbs:     []string{"delete", "get"},
			},
			expected: []rbac.PolicyRule{
				{
					APIGroups: []string{"apps"},
					Resources: []string{"pods", "namespaces"},
					Verbs:     []string{"delete", "get"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove non resource in clusterrole",
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
				Verbs:           []string{"delete"},
			},
			expected: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/v1"},
					Verbs:           []string{"delete"},
				},
			},
			wantErr: false,
		},
		{
			name: "remove non resource with wildcard permission in clusterrole",
			obj: &rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: "clusterrole",
				},
				Rules: []rbac.PolicyRule{
					{
						NonResourceURLs: []string{"/apis/*", "/apis/v1"},
						Verbs:           []string{"*"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"get", "post"},
			},
			wantErr: true,
		},
		{
			name: "remove wildcard permission non rerource from clusterrole",
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
				Verbs:           []string{"*"},
			},
			wantErr: true,
		},
		{
			name: "remove wildcard permission non rerource from clusterrole which have wildcard",
			obj: &rbac.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: "clusterrole",
				},
				Rules: []rbac.PolicyRule{
					{
						NonResourceURLs: []string{"/apis/*", "/apis/v1"},
						Verbs:           []string{"*"},
					},
				},
			},
			rule: &rbac.PolicyRule{
				NonResourceURLs: []string{"/apis/*"},
				Verbs:           []string{"*"},
			},
			expected: []rbac.PolicyRule{
				{
					NonResourceURLs: []string{"/apis/v1"},
					Verbs:           []string{"*"},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		if err := updatePolicyRuleForObject(tt.obj, tt.rule, true); (err != nil) != tt.wantErr {
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
		"test-duplicate-verbs": {
			ruleOptions: &RuleOptions{
				verbs: []string{
					"get",
					"watch",
					"list",
					"get",
				},
			},
			expected: &RuleOptions{
				verbs: []string{
					"get",
					"watch",
					"list",
				},
				resources: []ResourceOptions{
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
				resourceNames:   []string{},
				nonResourceURLs: []string{},
			},
			expectErr: false,
		},
		"test-verball": {
			ruleOptions: &RuleOptions{
				verbs: []string{
					"get",
					"watch",
					"list",
					"*",
				},
			},
			expected: &RuleOptions{
				verbs: []string{"*"},
				resources: []ResourceOptions{
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
				resourceNames:   []string{},
				nonResourceURLs: []string{},
			},
			expectErr: false,
		},
		"test-duplicate-resourcenames": {
			ruleOptions: &RuleOptions{
				verbs:         []string{"*"},
				resourceNames: []string{"foo", "foo"},
			},
			expected: &RuleOptions{
				verbs: []string{"*"},
				resources: []ResourceOptions{
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
				resourceNames:   []string{"foo"},
				nonResourceURLs: []string{},
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
		"test-duplicate-verbs": {
			ruleOptions: &RuleOptions{
				verbs: []string{
					"get",
					"put",
					"post",
					"get",
				},
				nonResourceURLs: []string{"test", "test"},
			},
			expected: &RuleOptions{
				verbs: []string{
					"get",
					"put",
					"post",
				},
				resources:       []ResourceOptions{},
				resourceNames:   []string{},
				nonResourceURLs: []string{"test"},
			},
			expectErr: false,
		},
		"test-verball": {
			ruleOptions: &RuleOptions{
				verbs: []string{
					"get",
					"put",
					"post",
					"*",
				},
				nonResourceURLs: []string{"test"},
			},
			expected: &RuleOptions{
				verbs:           []string{"*"},
				resources:       []ResourceOptions{},
				resourceNames:   []string{},
				nonResourceURLs: []string{"test"},
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
				verbs: []string{"get"},
			},
			expectErr: true,
		},
		"test-invalid-verb-for-resource": {
			ruleOptions: &RuleOptions{
				verbs: []string{"invalid-verb"},
				resources: []ResourceOptions{
					{
						Resource: "pods",
					},
				},
			},
			expectErr: true,
		},
		"test-invalid-for-nonresource": {
			ruleOptions: &RuleOptions{
				verbs:           []string{"list"},
				nonResourceURLs: []string{"test"},
			},
			expectErr: true,
		},
		"test-invalid-resource": {
			ruleOptions: &RuleOptions{
				verbs: []string{"get"},
				resources: []ResourceOptions{
					{
						Resource: "invalid-resource",
					},
				},
			},
			expectErr: true,
		},
		"test-invalid-subresource": {
			ruleOptions: &RuleOptions{
				verbs: []string{"get"},
				resources: []ResourceOptions{
					{
						Resource:    "replicasets",
						SubResource: "invalid-resource",
					},
				},
			},
			expectErr: true,
		},
		"test-resource-name-with-multiple-resources": {
			ruleOptions: &RuleOptions{
				verbs: []string{"get"},
				resources: []ResourceOptions{
					{
						Resource: "pods",
					},
					{
						Resource: "deployments",
						Group:    "extensions",
					},
				},
				resourceNames: []string{"foo"},
			},
			expectErr: true,
		},
		"test-valid-case": {
			ruleOptions: &RuleOptions{
				verbs: []string{"get", "list"},
				resources: []ResourceOptions{
					{
						Resource:    "replicasets",
						SubResource: "scales",
					},
				},
				resourceNames: []string{"foo"},
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
