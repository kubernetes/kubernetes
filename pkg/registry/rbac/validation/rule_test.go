/*
Copyright 2016 The Kubernetes Authors.

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

package validation

import (
	"hash/fnv"
	"io"
	"reflect"
	"sort"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// compute a hash of a policy rule so we can sort in a deterministic order
func hashOf(p rbac.PolicyRule) string {
	hash := fnv.New32()
	writeStrings := func(slis ...[]string) {
		for _, sli := range slis {
			for _, s := range sli {
				io.WriteString(hash, s)
			}
		}
	}
	writeStrings(p.Verbs, p.APIGroups, p.Resources, p.ResourceNames, p.NonResourceURLs)
	return string(hash.Sum(nil))
}

// byHash sorts a set of policy rules by a hash of its fields
type byHash []rbac.PolicyRule

func (b byHash) Len() int           { return len(b) }
func (b byHash) Less(i, j int) bool { return hashOf(b[i]) < hashOf(b[j]) }
func (b byHash) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }

func TestDefaultRuleResolver(t *testing.T) {
	ruleReadPods := rbac.PolicyRule{
		Verbs:     []string{"GET", "WATCH"},
		APIGroups: []string{"v1"},
		Resources: []string{"pods"},
	}
	ruleReadServices := rbac.PolicyRule{
		Verbs:     []string{"GET", "WATCH"},
		APIGroups: []string{"v1"},
		Resources: []string{"services"},
	}
	ruleWriteNodes := rbac.PolicyRule{
		Verbs:     []string{"PUT", "CREATE", "UPDATE"},
		APIGroups: []string{"v1"},
		Resources: []string{"nodes"},
	}
	ruleAdmin := rbac.PolicyRule{
		Verbs:     []string{"*"},
		APIGroups: []string{"*"},
		Resources: []string{"*"},
	}

	staticRoles1 := StaticRoles{
		roles: []*rbac.Role{
			{
				ObjectMeta: api.ObjectMeta{Namespace: "namespace1", Name: "readthings"},
				Rules:      []rbac.PolicyRule{ruleReadPods, ruleReadServices},
			},
		},
		clusterRoles: []*rbac.ClusterRole{
			{
				ObjectMeta: api.ObjectMeta{Name: "cluster-admin"},
				Rules:      []rbac.PolicyRule{ruleAdmin},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "write-nodes"},
				Rules:      []rbac.PolicyRule{ruleWriteNodes},
			},
		},
		roleBindings: []*rbac.RoleBinding{
			{
				ObjectMeta: api.ObjectMeta{Namespace: "namespace1"},
				Subjects: []rbac.Subject{
					{Kind: rbac.UserKind, Name: "foobar"},
					{Kind: rbac.GroupKind, Name: "group1"},
				},
				RoleRef: rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "Role", Name: "readthings"},
			},
		},
		clusterRoleBindings: []*rbac.ClusterRoleBinding{
			{
				Subjects: []rbac.Subject{
					{Kind: rbac.UserKind, Name: "admin"},
					{Kind: rbac.GroupKind, Name: "admin"},
				},
				RoleRef: rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "cluster-admin"},
			},
		},
	}

	tests := []struct {
		StaticRoles

		// For a given context, what are the rules that apply?
		user           user.Info
		namespace      string
		effectiveRules []rbac.PolicyRule
	}{
		{
			StaticRoles:    staticRoles1,
			user:           &user.DefaultInfo{Name: "foobar"},
			namespace:      "namespace1",
			effectiveRules: []rbac.PolicyRule{ruleReadPods, ruleReadServices},
		},
		{
			StaticRoles:    staticRoles1,
			user:           &user.DefaultInfo{Name: "foobar"},
			namespace:      "namespace2",
			effectiveRules: []rbac.PolicyRule{},
		},
		{
			StaticRoles: staticRoles1,
			// Same as above but without a namespace. Only cluster rules should apply.
			user:           &user.DefaultInfo{Name: "foobar", Groups: []string{"admin"}},
			effectiveRules: []rbac.PolicyRule{ruleAdmin},
		},
		{
			StaticRoles:    staticRoles1,
			user:           &user.DefaultInfo{},
			effectiveRules: []rbac.PolicyRule{},
		},
	}

	for i, tc := range tests {
		ruleResolver := newMockRuleResolver(&tc.StaticRoles)
		rules, err := ruleResolver.RulesFor(tc.user, tc.namespace)
		if err != nil {
			t.Errorf("case %d: GetEffectivePolicyRules(context)=%v", i, err)
			continue
		}

		// Sort for deep equals
		sort.Sort(byHash(rules))
		sort.Sort(byHash(tc.effectiveRules))

		if !reflect.DeepEqual(rules, tc.effectiveRules) {
			ruleDiff := diff.ObjectDiff(rules, tc.effectiveRules)
			t.Errorf("case %d: %s", i, ruleDiff)
		}
	}
}

func TestAppliesTo(t *testing.T) {
	tests := []struct {
		subjects  []rbac.Subject
		user      user.Info
		namespace string
		appliesTo bool
		testCase  string
	}{
		{
			subjects: []rbac.Subject{
				{Kind: rbac.UserKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "foobar"},
			appliesTo: true,
			testCase:  "single subject that matches username",
		},
		{
			subjects: []rbac.Subject{
				{Kind: rbac.UserKind, Name: "barfoo"},
				{Kind: rbac.UserKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "foobar"},
			appliesTo: true,
			testCase:  "multiple subjects, one that matches username",
		},
		{
			subjects: []rbac.Subject{
				{Kind: rbac.UserKind, Name: "barfoo"},
				{Kind: rbac.UserKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "zimzam"},
			appliesTo: false,
			testCase:  "multiple subjects, none that match username",
		},
		{
			subjects: []rbac.Subject{
				{Kind: rbac.UserKind, Name: "barfoo"},
				{Kind: rbac.GroupKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "zimzam", Groups: []string{"foobar"}},
			appliesTo: true,
			testCase:  "multiple subjects, one that match group",
		},
		{
			subjects: []rbac.Subject{
				{Kind: rbac.UserKind, Name: "barfoo"},
				{Kind: rbac.GroupKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "zimzam", Groups: []string{"foobar"}},
			namespace: "namespace1",
			appliesTo: true,
			testCase:  "multiple subjects, one that match group, should ignore namespace",
		},
		{
			subjects: []rbac.Subject{
				{Kind: rbac.UserKind, Name: "barfoo"},
				{Kind: rbac.GroupKind, Name: "foobar"},
				{Kind: rbac.ServiceAccountKind, Namespace: "kube-system", Name: "default"},
			},
			user:      &user.DefaultInfo{Name: "system:serviceaccount:kube-system:default"},
			namespace: "default",
			appliesTo: true,
			testCase:  "multiple subjects with a service account that matches",
		},
		{
			subjects: []rbac.Subject{
				{Kind: rbac.UserKind, Name: "*"},
			},
			user:      &user.DefaultInfo{Name: "foobar"},
			namespace: "default",
			appliesTo: false,
			testCase:  "* user subject name doesn't match all users",
		},
		{
			subjects: []rbac.Subject{
				{Kind: rbac.GroupKind, Name: user.AllAuthenticated},
				{Kind: rbac.GroupKind, Name: user.AllUnauthenticated},
			},
			user:      &user.DefaultInfo{Name: "foobar", Groups: []string{user.AllAuthenticated}},
			namespace: "default",
			appliesTo: true,
			testCase:  "binding to all authenticated and unauthenticated subjects matches authenticated user",
		},
		{
			subjects: []rbac.Subject{
				{Kind: rbac.GroupKind, Name: user.AllAuthenticated},
				{Kind: rbac.GroupKind, Name: user.AllUnauthenticated},
			},
			user:      &user.DefaultInfo{Name: "system:anonymous", Groups: []string{user.AllUnauthenticated}},
			namespace: "default",
			appliesTo: true,
			testCase:  "binding to all authenticated and unauthenticated subjects matches anonymous user",
		},
	}

	for _, tc := range tests {
		got := appliesTo(tc.user, tc.subjects, tc.namespace)
		if got != tc.appliesTo {
			t.Errorf("case %q want appliesTo=%t, got appliesTo=%t", tc.testCase, tc.appliesTo, got)
		}
	}
}
