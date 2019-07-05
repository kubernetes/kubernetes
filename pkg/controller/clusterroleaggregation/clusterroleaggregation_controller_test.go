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

package clusterroleaggregation

import (
	"testing"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	rbaclisters "k8s.io/client-go/listers/rbac/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller"
)

func TestSyncClusterRole(t *testing.T) {
	hammerRules := func() []rbacv1.PolicyRule {
		return []rbacv1.PolicyRule{
			{Verbs: []string{"hammer"}, Resources: []string{"nails"}},
			{Verbs: []string{"hammer"}, Resources: []string{"wedges"}},
		}
	}
	chiselRules := func() []rbacv1.PolicyRule {
		return []rbacv1.PolicyRule{
			{Verbs: []string{"chisel"}, Resources: []string{"mortises"}},
		}
	}
	sawRules := func() []rbacv1.PolicyRule {
		return []rbacv1.PolicyRule{
			{Verbs: []string{"saw"}, Resources: []string{"boards"}},
		}
	}
	role := func(name string, labels map[string]string, rules []rbacv1.PolicyRule) *rbacv1.ClusterRole {
		return &rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels},
			Rules:      rules,
		}
	}
	combinedRole := func(selectors []map[string]string, rules ...[]rbacv1.PolicyRule) *rbacv1.ClusterRole {
		ret := &rbacv1.ClusterRole{
			ObjectMeta:      metav1.ObjectMeta{Name: "combined"},
			AggregationRule: &rbacv1.AggregationRule{},
		}
		for _, selector := range selectors {
			ret.AggregationRule.ClusterRoleSelectors = append(ret.AggregationRule.ClusterRoleSelectors,
				metav1.LabelSelector{MatchLabels: selector})
		}
		for _, currRules := range rules {
			ret.Rules = append(ret.Rules, currRules...)
		}
		return ret
	}

	tests := []struct {
		name                 string
		startingClusterRoles []*rbacv1.ClusterRole
		clusterRoleToSync    string
		expectedClusterRole  *rbacv1.ClusterRole
	}{
		{
			name: "remove dead rules",
			startingClusterRoles: []*rbacv1.ClusterRole{
				role("hammer", map[string]string{"foo": "bar"}, hammerRules()),
				combinedRole([]map[string]string{{"foo": "bar"}}, sawRules()),
			},
			clusterRoleToSync:   "combined",
			expectedClusterRole: combinedRole([]map[string]string{{"foo": "bar"}}, hammerRules()),
		},
		{
			name: "strip rules",
			startingClusterRoles: []*rbacv1.ClusterRole{
				role("hammer", map[string]string{"foo": "not-bar"}, hammerRules()),
				combinedRole([]map[string]string{{"foo": "bar"}}, hammerRules()),
			},
			clusterRoleToSync:   "combined",
			expectedClusterRole: combinedRole([]map[string]string{{"foo": "bar"}}),
		},
		{
			name: "select properly and put in order",
			startingClusterRoles: []*rbacv1.ClusterRole{
				role("hammer", map[string]string{"foo": "bar"}, hammerRules()),
				role("chisel", map[string]string{"foo": "bar"}, chiselRules()),
				role("saw", map[string]string{"foo": "not-bar"}, sawRules()),
				combinedRole([]map[string]string{{"foo": "bar"}}),
			},
			clusterRoleToSync:   "combined",
			expectedClusterRole: combinedRole([]map[string]string{{"foo": "bar"}}, chiselRules(), hammerRules()),
		},
		{
			name: "select properly with multiple selectors",
			startingClusterRoles: []*rbacv1.ClusterRole{
				role("hammer", map[string]string{"foo": "bar"}, hammerRules()),
				role("chisel", map[string]string{"foo": "bar"}, chiselRules()),
				role("saw", map[string]string{"foo": "not-bar"}, sawRules()),
				combinedRole([]map[string]string{{"foo": "bar"}, {"foo": "not-bar"}}),
			},
			clusterRoleToSync:   "combined",
			expectedClusterRole: combinedRole([]map[string]string{{"foo": "bar"}, {"foo": "not-bar"}}, chiselRules(), hammerRules(), sawRules()),
		},
		{
			name: "select properly remove duplicates",
			startingClusterRoles: []*rbacv1.ClusterRole{
				role("hammer", map[string]string{"foo": "bar"}, hammerRules()),
				role("chisel", map[string]string{"foo": "bar"}, chiselRules()),
				role("saw", map[string]string{"foo": "bar"}, sawRules()),
				role("other-saw", map[string]string{"foo": "not-bar"}, sawRules()),
				combinedRole([]map[string]string{{"foo": "bar"}, {"foo": "not-bar"}}),
			},
			clusterRoleToSync:   "combined",
			expectedClusterRole: combinedRole([]map[string]string{{"foo": "bar"}, {"foo": "not-bar"}}, chiselRules(), hammerRules(), sawRules()),
		},
		{
			name: "no diff skip",
			startingClusterRoles: []*rbacv1.ClusterRole{
				role("hammer", map[string]string{"foo": "bar"}, hammerRules()),
				combinedRole([]map[string]string{{"foo": "bar"}}, hammerRules()),
			},
			clusterRoleToSync:   "combined",
			expectedClusterRole: nil,
		}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			indexer := cache.NewIndexer(controller.KeyFunc, cache.Indexers{})
			objs := []runtime.Object{}
			for _, obj := range test.startingClusterRoles {
				objs = append(objs, obj)
				indexer.Add(obj)
			}
			fakeClient := fakeclient.NewSimpleClientset(objs...)
			c := ClusterRoleAggregationController{
				clusterRoleClient: fakeClient.RbacV1(),
				clusterRoleLister: rbaclisters.NewClusterRoleLister(indexer),
			}
			err := c.syncClusterRole(test.clusterRoleToSync)
			if err != nil {
				t.Fatal(err)
			}

			if test.expectedClusterRole == nil {
				if len(fakeClient.Actions()) != 0 {
					t.Fatalf("unexpected actions %#v", fakeClient.Actions())
				}
				return
			}
			if len(fakeClient.Actions()) != 1 {
				t.Fatalf("unexpected actions %#v", fakeClient.Actions())
			}

			action := fakeClient.Actions()[0]
			if !action.Matches("update", "clusterroles") {
				t.Fatalf("unexpected action %#v", action)
			}
			updateAction, ok := action.(clienttesting.UpdateAction)
			if !ok {
				t.Fatalf("unexpected action %#v", action)
			}
			if !equality.Semantic.DeepEqual(updateAction.GetObject().(*rbacv1.ClusterRole), test.expectedClusterRole) {
				t.Fatalf("%v", diff.ObjectDiff(test.expectedClusterRole, updateAction.GetObject().(*rbacv1.ClusterRole)))

			}
		})
	}
}
