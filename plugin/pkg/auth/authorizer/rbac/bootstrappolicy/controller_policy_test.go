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
	"testing"

	"k8s.io/kubernetes/pkg/util/sets"
)

// rolesWithAllowStar are the controller roles which are allowed to contain a *.  These are
// namespace lifecycle and GC which have to delete anything.  If you're adding to this list
// tag sig-auth
var rolesWithAllowStar = sets.NewString(
	saRolePrefix+"namespace-controller",
	saRolePrefix+"generic-garbage-collector",
)

// TestNoStarsForControllers confirms that no controller role has star verbs, groups,
// or resources.  There are two known exceptions, namespace lifecycle and GC which have to
// delete anything
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
