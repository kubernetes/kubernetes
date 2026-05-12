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

	"github.com/google/go-cmp/cmp"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// compute a hash of a policy rule so we can sort in a deterministic order
func hashOf(p rbacv1.PolicyRule) string {
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
type byHash []rbacv1.PolicyRule

func (b byHash) Len() int           { return len(b) }
func (b byHash) Less(i, j int) bool { return hashOf(b[i]) < hashOf(b[j]) }
func (b byHash) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }

func TestDefaultRuleResolver(t *testing.T) {
	ruleReadPods := rbacv1.PolicyRule{
		Verbs:     []string{"GET", "WATCH"},
		APIGroups: []string{"v1"},
		Resources: []string{"pods"},
	}
	ruleReadServices := rbacv1.PolicyRule{
		Verbs:     []string{"GET", "WATCH"},
		APIGroups: []string{"v1"},
		Resources: []string{"services"},
	}
	ruleWriteNodes := rbacv1.PolicyRule{
		Verbs:     []string{"PUT", "CREATE", "UPDATE"},
		APIGroups: []string{"v1"},
		Resources: []string{"nodes"},
	}
	ruleAdmin := rbacv1.PolicyRule{
		Verbs:     []string{"*"},
		APIGroups: []string{"*"},
		Resources: []string{"*"},
	}

	staticRoles1 := StaticRoles{
		roles: []*rbacv1.Role{
			{
				ObjectMeta: metav1.ObjectMeta{Namespace: "namespace1", Name: "readthings"},
				Rules:      []rbacv1.PolicyRule{ruleReadPods, ruleReadServices},
			},
		},
		clusterRoles: []*rbacv1.ClusterRole{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "cluster-admin"},
				Rules:      []rbacv1.PolicyRule{ruleAdmin},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "write-nodes"},
				Rules:      []rbacv1.PolicyRule{ruleWriteNodes},
			},
		},
		roleBindings: []*rbacv1.RoleBinding{
			{
				ObjectMeta: metav1.ObjectMeta{Namespace: "namespace1"},
				Subjects: []rbacv1.Subject{
					{Kind: rbacv1.UserKind, Name: "foobar"},
					{Kind: rbacv1.GroupKind, Name: "group1"},
				},
				RoleRef: rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "readthings"},
			},
		},
		clusterRoleBindings: []*rbacv1.ClusterRoleBinding{
			{
				Subjects: []rbacv1.Subject{
					{Kind: rbacv1.UserKind, Name: "admin"},
					{Kind: rbacv1.GroupKind, Name: "admin"},
				},
				RoleRef: rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "ClusterRole", Name: "cluster-admin"},
			},
		},
	}

	tests := []struct {
		StaticRoles

		// For a given context, what are the rules that apply?
		user           user.Info
		namespace      string
		effectiveRules []rbacv1.PolicyRule
	}{
		{
			StaticRoles:    staticRoles1,
			user:           &user.DefaultInfo{Name: "foobar"},
			namespace:      "namespace1",
			effectiveRules: []rbacv1.PolicyRule{ruleReadPods, ruleReadServices},
		},
		{
			StaticRoles:    staticRoles1,
			user:           &user.DefaultInfo{Name: "foobar"},
			namespace:      "namespace2",
			effectiveRules: nil,
		},
		{
			StaticRoles: staticRoles1,
			// Same as above but without a namespace. Only cluster rules should apply.
			user:           &user.DefaultInfo{Name: "foobar", Groups: []string{"admin"}},
			effectiveRules: []rbacv1.PolicyRule{ruleAdmin},
		},
		{
			StaticRoles:    staticRoles1,
			user:           &user.DefaultInfo{},
			effectiveRules: nil,
		},
	}

	for i, tc := range tests {
		ruleResolver := newMockRuleResolver(&tc.StaticRoles)
		rules, err := ruleResolver.RulesFor(genericapirequest.NewContext(), tc.user, tc.namespace)
		if err != nil {
			t.Errorf("case %d: GetEffectivePolicyRules(context)=%v", i, err)
			continue
		}

		// Sort for deep equals
		sort.Sort(byHash(rules))
		sort.Sort(byHash(tc.effectiveRules))

		if !reflect.DeepEqual(rules, tc.effectiveRules) {
			ruleDiff := cmp.Diff(rules, tc.effectiveRules)
			t.Errorf("case %d: %s", i, ruleDiff)
		}
	}
}

func TestAppliesTo(t *testing.T) {
	tests := []struct {
		subjects  []rbacv1.Subject
		user      user.Info
		namespace string
		appliesTo bool
		index     int
		testCase  string
	}{
		{
			subjects: []rbacv1.Subject{
				{Kind: rbacv1.UserKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "foobar"},
			appliesTo: true,
			index:     0,
			testCase:  "single subject that matches username",
		},
		{
			subjects: []rbacv1.Subject{
				{Kind: rbacv1.UserKind, Name: "barfoo"},
				{Kind: rbacv1.UserKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "foobar"},
			appliesTo: true,
			index:     1,
			testCase:  "multiple subjects, one that matches username",
		},
		{
			subjects: []rbacv1.Subject{
				{Kind: rbacv1.UserKind, Name: "barfoo"},
				{Kind: rbacv1.UserKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "zimzam"},
			appliesTo: false,
			testCase:  "multiple subjects, none that match username",
		},
		{
			subjects: []rbacv1.Subject{
				{Kind: rbacv1.UserKind, Name: "barfoo"},
				{Kind: rbacv1.GroupKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "zimzam", Groups: []string{"foobar"}},
			appliesTo: true,
			index:     1,
			testCase:  "multiple subjects, one that match group",
		},
		{
			subjects: []rbacv1.Subject{
				{Kind: rbacv1.UserKind, Name: "barfoo"},
				{Kind: rbacv1.GroupKind, Name: "foobar"},
			},
			user:      &user.DefaultInfo{Name: "zimzam", Groups: []string{"foobar"}},
			namespace: "namespace1",
			appliesTo: true,
			index:     1,
			testCase:  "multiple subjects, one that match group, should ignore namespace",
		},
		{
			subjects: []rbacv1.Subject{
				{Kind: rbacv1.UserKind, Name: "barfoo"},
				{Kind: rbacv1.GroupKind, Name: "foobar"},
				{Kind: rbacv1.ServiceAccountKind, Namespace: "kube-system", Name: "default"},
			},
			user:      &user.DefaultInfo{Name: "system:serviceaccount:kube-system:default"},
			namespace: "default",
			appliesTo: true,
			index:     2,
			testCase:  "multiple subjects with a service account that matches",
		},
		{
			subjects: []rbacv1.Subject{
				{Kind: rbacv1.UserKind, Name: "*"},
			},
			user:      &user.DefaultInfo{Name: "foobar"},
			namespace: "default",
			appliesTo: false,
			testCase:  "* user subject name doesn't match all users",
		},
		{
			subjects: []rbacv1.Subject{
				{Kind: rbacv1.GroupKind, Name: user.AllAuthenticated},
				{Kind: rbacv1.GroupKind, Name: user.AllUnauthenticated},
			},
			user:      &user.DefaultInfo{Name: "foobar", Groups: []string{user.AllAuthenticated}},
			namespace: "default",
			appliesTo: true,
			index:     0,
			testCase:  "binding to all authenticated and unauthenticated subjects matches authenticated user",
		},
		{
			subjects: []rbacv1.Subject{
				{Kind: rbacv1.GroupKind, Name: user.AllAuthenticated},
				{Kind: rbacv1.GroupKind, Name: user.AllUnauthenticated},
			},
			user:      &user.DefaultInfo{Name: "system:anonymous", Groups: []string{user.AllUnauthenticated}},
			namespace: "default",
			appliesTo: true,
			index:     1,
			testCase:  "binding to all authenticated and unauthenticated subjects matches anonymous user",
		},
	}

	for _, tc := range tests {
		gotIndex, got := appliesTo(tc.user, tc.subjects, tc.namespace)
		if got != tc.appliesTo {
			t.Errorf("case %q want appliesTo=%t, got appliesTo=%t", tc.testCase, tc.appliesTo, got)
		}
		if gotIndex != tc.index {
			t.Errorf("case %q want index %d, got %d", tc.testCase, tc.index, gotIndex)
		}
	}
}
