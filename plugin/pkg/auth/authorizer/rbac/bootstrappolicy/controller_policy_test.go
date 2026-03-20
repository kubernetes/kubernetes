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

package bootstrappolicy

import (
	"reflect"
	"slices"
	"testing"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

// rolesWithAllowStar are the controller roles which are allowed to contain a *.  These are
// namespace lifecycle and GC which have to delete anything.  If you're adding to this list
// tag sig-auth
var rolesWithAllowStar = sets.NewString(
	saRolePrefix+"namespace-controller",
	saRolePrefix+"generic-garbage-collector",
	saRolePrefix+"resourcequota-controller",
	saRolePrefix+"horizontal-pod-autoscaler",
	saRolePrefix+"clusterrole-aggregation-controller",
	saRolePrefix+"disruption-controller",
)

// TestNoStarsForControllers confirms that no controller role has star verbs, groups,
// or resources. There are three known exceptions: namespace lifecycle and GC which have to
// delete anything, and HPA, which has the power to read metrics associated
// with any object.
func TestNoStarsForControllers(t *testing.T) {
	for _, role := range ControllerRoles() {
		if rolesWithAllowStar.Has(role.Name) {
			continue
		}

		for i, rule := range role.Rules {
			for j, verb := range rule.Verbs {
				if verb == "*" {
					t.Errorf("%s.Rule[%d].Verbs[%d] is star", role.Name, i, j)
				}
			}
			for j, group := range rule.APIGroups {
				if group == "*" {
					t.Errorf("%s.Rule[%d].APIGroups[%d] is star", role.Name, i, j)
				}
			}
			for j, resource := range rule.Resources {
				if resource == "*" {
					t.Errorf("%s.Rule[%d].Resources[%d] is star", role.Name, i, j)
				}
			}
		}
	}
}

func TestControllerRoleLabel(t *testing.T) {
	roles := ControllerRoles()
	for i := range roles {
		role := roles[i]
		accessor, err := meta.Accessor(&role)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got, want := accessor.GetLabels(), map[string]string{"kubernetes.io/bootstrapping": "rbac-defaults"}; !reflect.DeepEqual(got, want) {
			t.Errorf("ClusterRole: %s GetLabels() = %s, want %s", accessor.GetName(), got, want)
		}
	}

	rolebindings := ControllerRoleBindings()
	for i := range rolebindings {
		rolebinding := rolebindings[i]
		accessor, err := meta.Accessor(&rolebinding)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got, want := accessor.GetLabels(), map[string]string{"kubernetes.io/bootstrapping": "rbac-defaults"}; !reflect.DeepEqual(got, want) {
			t.Errorf("ClusterRoleBinding: %s GetLabels() = %s, want %s", accessor.GetName(), got, want)
		}
	}
}

func TestPodGroupProtectionControllerRBAC(t *testing.T) {
	roleName := saRolePrefix + "podgroup-protection-controller"

	tests := []struct {
		name              string
		enableFeatureGate bool
		expectRole        bool
	}{
		{
			name:              "role and binding absent when GenericWorkload is disabled",
			enableFeatureGate: false,
			expectRole:        false,
		},
		{
			name:              "role and binding present when GenericWorkload is enabled",
			enableFeatureGate: true,
			expectRole:        true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, test.enableFeatureGate)

			var foundRole *rbacv1.ClusterRole
			for i, role := range ControllerRoles() {
				if role.Name == roleName {
					foundRole = &ControllerRoles()[i]
					break
				}
			}

			if !test.expectRole {
				if foundRole != nil {
					t.Fatalf("role %q should not exist when GenericWorkload is disabled", roleName)
				}
				return
			}

			if foundRole == nil {
				t.Fatalf("role %q not found when GenericWorkload is enabled", roleName)
			}

			wantRules := []rbacv1.PolicyRule{
				{Verbs: []string{"get", "list", "update", "watch"}, APIGroups: []string{"scheduling.k8s.io"}, Resources: []string{"podgroups"}},
				{Verbs: []string{"get", "list", "watch"}, APIGroups: []string{""}, Resources: []string{"pods"}},
			}
			if !reflect.DeepEqual(foundRole.Rules, wantRules) {
				t.Errorf("unexpected rules:\ngot:  %+v\nwant: %+v", foundRole.Rules, wantRules)
			}

			var foundBinding *rbacv1.ClusterRoleBinding
			for i, binding := range ControllerRoleBindings() {
				if binding.RoleRef.Name == roleName {
					foundBinding = &ControllerRoleBindings()[i]
					break
				}
			}
			if foundBinding == nil {
				t.Fatalf("binding for %q not found", roleName)
			}
			if len(foundBinding.Subjects) != 1 {
				t.Fatalf("expected 1 subject, got %d", len(foundBinding.Subjects))
			}
			if got, want := foundBinding.Subjects[0].Name, "podgroup-protection-controller"; got != want {
				t.Errorf("subject name = %q, want %q", got, want)
			}
			if got, want := foundBinding.Subjects[0].Namespace, "kube-system"; got != want {
				t.Errorf("subject namespace = %q, want %q", got, want)
			}
		})
	}
}

func TestControllerRoleVerbsConsistency(t *testing.T) {
	roles := ControllerRoles()
	for _, role := range roles {
		for _, rule := range role.Rules {
			verbs := rule.Verbs
			if slices.Contains(verbs, "list") && !slices.Contains(verbs, "watch") {
				t.Errorf("The ClusterRole %s has Verb `List` but does not have Verb `Watch`.", role.Name)
			}
		}
	}
}

func TestJobControllerSchedulingRules(t *testing.T) {
	cases := []struct {
		name                  string
		enableWorkloadWithJob bool
		wantSchedulingRules   bool
	}{
		{
			name:                  "feature gate disabled",
			enableWorkloadWithJob: false,
			wantSchedulingRules:   false,
		},
		{
			name:                  "feature gate enabled",
			enableWorkloadWithJob: true,
			wantSchedulingRules:   true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:       tc.enableWorkloadWithJob,
				features.EnableWorkloadWithJob: tc.enableWorkloadWithJob,
			})
			got := hasSchedulingRules(ControllerRoles())
			if got != tc.wantSchedulingRules {
				t.Errorf("hasSchedulingRules() = %v, want %v", got, tc.wantSchedulingRules)
			}
		})
	}
}

func hasSchedulingRules(roles []rbacv1.ClusterRole) bool {
	for _, role := range roles {
		if role.Name != saRolePrefix+"job-controller" {
			continue
		}
		for _, rule := range role.Rules {
			if slices.Contains(rule.APIGroups, "scheduling.k8s.io") &&
				slices.Contains(rule.Resources, "workloads") &&
				slices.Contains(rule.Resources, "podgroups") {
				return true
			}
		}
	}
	return false
}
