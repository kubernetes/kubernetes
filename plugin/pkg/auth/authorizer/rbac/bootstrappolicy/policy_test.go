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

package bootstrappolicy_test

import (
	"testing"

	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	rbacvalidation "k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

// semanticRoles is a few enumerated roles for which the relationships are well established
// and we want to maintain symmetric roles
type semanticRoles struct {
	admin *rbac.ClusterRole
	edit  *rbac.ClusterRole
	view  *rbac.ClusterRole
}

func getSemanticRoles(roles []rbac.ClusterRole) semanticRoles {
	ret := semanticRoles{}
	for i := range roles {
		role := roles[i]
		switch role.Name {
		case "admin":
			ret.admin = &role
		case "edit":
			ret.edit = &role
		case "view":
			ret.view = &role
		}
	}
	return ret
}

// Some roles should always cover others
func TestCovers(t *testing.T) {
	semanticRoles := getSemanticRoles(bootstrappolicy.ClusterRoles())

	if covers, miss := rbacvalidation.Covers(semanticRoles.admin.Rules, semanticRoles.edit.Rules); !covers {
		t.Errorf("failed to cover: %#v", miss)
	}
	if covers, miss := rbacvalidation.Covers(semanticRoles.admin.Rules, semanticRoles.view.Rules); !covers {
		t.Errorf("failed to cover: %#v", miss)
	}
	if covers, miss := rbacvalidation.Covers(semanticRoles.edit.Rules, semanticRoles.view.Rules); !covers {
		t.Errorf("failed to cover: %#v", miss)
	}
}

// additionalAdminPowers is the list of powers that we expect to be different than the editor role.
// one resource per rule to make the "does not already contain" check easy
var additionalAdminPowers = []rbac.PolicyRule{
	rbac.NewRule("create").Groups("authorization.k8s.io").Resources("localsubjectaccessreviews").RuleOrDie(),
	rbac.NewRule(bootstrappolicy.ReadWrite...).Groups("rbac.authorization.k8s.io").Resources("rolebindings").RuleOrDie(),
	rbac.NewRule(bootstrappolicy.ReadWrite...).Groups("rbac.authorization.k8s.io").Resources("roles").RuleOrDie(),
}

func TestAdminEditRelationship(t *testing.T) {
	semanticRoles := getSemanticRoles(bootstrappolicy.ClusterRoles())

	// confirm that the edit role doesn't already have extra powers
	for _, rule := range additionalAdminPowers {
		if covers, _ := rbacvalidation.Covers(semanticRoles.edit.Rules, []rbac.PolicyRule{rule}); covers {
			t.Errorf("edit has extra powers: %#v", rule)
		}
	}
	semanticRoles.edit.Rules = append(semanticRoles.edit.Rules, additionalAdminPowers...)

	// at this point, we should have a two way covers relationship
	if covers, miss := rbacvalidation.Covers(semanticRoles.admin.Rules, semanticRoles.edit.Rules); !covers {
		t.Errorf("admin has lost rules for: %#v", miss)
	}
	if covers, miss := rbacvalidation.Covers(semanticRoles.edit.Rules, semanticRoles.admin.Rules); !covers {
		t.Errorf("edit is missing rules for: %#v\nIf these should only be admin powers, add them to the list.  Otherwise, add them to the edit role.", miss)
	}
}

// viewEscalatingNamespaceResources is the list of rules that would allow privilege escalation attacks based on
// ability to view (GET) them
var viewEscalatingNamespaceResources = []rbac.PolicyRule{
	rbac.NewRule(bootstrappolicy.Read...).Groups("").Resources("pods/attach").RuleOrDie(),
	rbac.NewRule(bootstrappolicy.Read...).Groups("").Resources("pods/proxy").RuleOrDie(),
	rbac.NewRule(bootstrappolicy.Read...).Groups("").Resources("pods/exec").RuleOrDie(),
	rbac.NewRule(bootstrappolicy.Read...).Groups("").Resources("pods/portforward").RuleOrDie(),
	rbac.NewRule(bootstrappolicy.Read...).Groups("").Resources("secrets").RuleOrDie(),
	rbac.NewRule(bootstrappolicy.Read...).Groups("").Resources("services/proxy").RuleOrDie(),
}

func TestEditViewRelationship(t *testing.T) {
	readVerbs := sets.NewString(bootstrappolicy.Read...)
	semanticRoles := getSemanticRoles(bootstrappolicy.ClusterRoles())

	// modify the edit role rules to make then read-only for comparison against view role rules
	for i := range semanticRoles.edit.Rules {
		rule := semanticRoles.edit.Rules[i]
		remainingVerbs := []string{}
		for _, verb := range rule.Verbs {
			if readVerbs.Has(verb) {
				remainingVerbs = append(remainingVerbs, verb)
			}
		}
		rule.Verbs = remainingVerbs
		semanticRoles.edit.Rules[i] = rule
	}

	// confirm that the view role doesn't already have extra powers
	for _, rule := range viewEscalatingNamespaceResources {
		if covers, _ := rbacvalidation.Covers(semanticRoles.view.Rules, []rbac.PolicyRule{rule}); covers {
			t.Errorf("view has extra powers: %#v", rule)
		}
	}
	semanticRoles.view.Rules = append(semanticRoles.view.Rules, viewEscalatingNamespaceResources...)

	// at this point, we should have a two way covers relationship
	if covers, miss := rbacvalidation.Covers(semanticRoles.edit.Rules, semanticRoles.view.Rules); !covers {
		t.Errorf("edit has lost rules for: %#v", miss)
	}
	if covers, miss := rbacvalidation.Covers(semanticRoles.view.Rules, semanticRoles.edit.Rules); !covers {
		t.Errorf("view is missing rules for: %#v\nIf these are escalating powers, add them to the list.  Otherwise, add them to the view role.", miss)
	}
}
