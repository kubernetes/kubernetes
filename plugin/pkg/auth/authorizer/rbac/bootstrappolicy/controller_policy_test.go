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
// or resources.  There are three known exceptions: namespace lifecycle and GC which have to
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

func TestJobControllerWorkloadsPermission(t *testing.T) {
	hasWorkloadsPermission := func(roles []rbacv1.ClusterRole) bool {
		for _, role := range roles {
			if role.Name != saRolePrefix+"job-controller" {
				continue
			}
			for _, rule := range role.Rules {
				if slices.Contains(rule.APIGroups, schedulingGroup) &&
					slices.Contains(rule.Resources, "workloads") {
					return true
				}
			}
		}
		return false
	}

	t.Run("feature gate disabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobGangPolicy, false)
		roles, _ := buildControllerRoles()
		if hasWorkloadsPermission(roles) {
			t.Errorf("job-controller should not have workloads permission when JobGangPolicy feature gate is disabled")
		}
	})

	t.Run("feature gate enabled", func(t *testing.T) {
		// JobGangPolicy depends on GangScheduling and GenericWorkload feature gates
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.GangScheduling:  true,
			features.GenericWorkload: true,
			features.JobGangPolicy:   true,
		})
		roles, _ := buildControllerRoles()
		if !hasWorkloadsPermission(roles) {
			t.Errorf("job-controller should have workloads permission when JobGangPolicy feature gate is enabled")
		}
	})
}
