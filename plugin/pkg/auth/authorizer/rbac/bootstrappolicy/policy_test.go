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
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/yaml"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-helpers/auth/rbac/validation"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

// semanticRoles is a few enumerated roles for which the relationships are well established
// and we want to maintain symmetric roles
type semanticRoles struct {
	admin *rbacv1.ClusterRole
	edit  *rbacv1.ClusterRole
	view  *rbacv1.ClusterRole
}

func getSemanticRoles(roles []rbacv1.ClusterRole) semanticRoles {
	ret := semanticRoles{}
	for i := range roles {
		role := roles[i]
		switch role.Name {
		case "system:aggregate-to-admin":
			ret.admin = &role
		case "system:aggregate-to-edit":
			ret.edit = &role
		case "system:aggregate-to-view":
			ret.view = &role
		}
	}
	return ret
}

// viewEscalatingNamespaceResources is the list of rules that would allow privilege escalation attacks based on
// ability to view (GET) them
var viewEscalatingNamespaceResources = []rbacv1.PolicyRule{
	rbacv1helpers.NewRule(bootstrappolicy.Read...).Groups("").Resources("pods/attach").RuleOrDie(),
	rbacv1helpers.NewRule(bootstrappolicy.Read...).Groups("").Resources("pods/proxy").RuleOrDie(),
	rbacv1helpers.NewRule(bootstrappolicy.Read...).Groups("").Resources("pods/exec").RuleOrDie(),
	rbacv1helpers.NewRule(bootstrappolicy.Read...).Groups("").Resources("pods/portforward").RuleOrDie(),
	rbacv1helpers.NewRule(bootstrappolicy.Read...).Groups("").Resources("secrets").RuleOrDie(),
	rbacv1helpers.NewRule(bootstrappolicy.Read...).Groups("").Resources("services/proxy").RuleOrDie(),
}

// ungettableResources is the list of rules that don't allow to view (GET) them
// this is purposefully separate list to distinguish from escalating privs
var ungettableResources = []rbacv1.PolicyRule{
	rbacv1helpers.NewRule(bootstrappolicy.Read...).Groups("apps", "extensions").Resources("deployments/rollback").RuleOrDie(),
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
		if covers, _ := validation.Covers(semanticRoles.view.Rules, []rbacv1.PolicyRule{rule}); covers {
			t.Errorf("view has extra powers: %#v", rule)
		}
	}
	semanticRoles.view.Rules = append(semanticRoles.view.Rules, viewEscalatingNamespaceResources...)

	// confirm that the view role doesn't have ungettable resources
	for _, rule := range ungettableResources {
		if covers, _ := validation.Covers(semanticRoles.view.Rules, []rbacv1.PolicyRule{rule}); covers {
			t.Errorf("view has ungettable resource: %#v", rule)
		}
	}
	semanticRoles.view.Rules = append(semanticRoles.view.Rules, ungettableResources...)
}

func TestBootstrapNamespaceRoles(t *testing.T) {
	list := &api.List{}
	names := sets.NewString()
	roles := map[string]runtime.Object{}

	namespaceRoles := bootstrappolicy.NamespaceRoles()
	for _, namespace := range sets.StringKeySet(namespaceRoles).List() {
		bootstrapRoles := namespaceRoles[namespace]
		for i := range bootstrapRoles {
			role := bootstrapRoles[i]
			names.Insert(role.Name)
			roles[role.Name] = &role
		}

		for _, name := range names.List() {
			list.Items = append(list.Items, roles[name])
		}
	}

	testObjects(t, list, "namespace-roles.yaml")
}

func TestBootstrapNamespaceRoleBindings(t *testing.T) {
	list := &api.List{}
	names := sets.NewString()
	roleBindings := map[string]runtime.Object{}

	namespaceRoleBindings := bootstrappolicy.NamespaceRoleBindings()
	for _, namespace := range sets.StringKeySet(namespaceRoleBindings).List() {
		bootstrapRoleBindings := namespaceRoleBindings[namespace]
		for i := range bootstrapRoleBindings {
			roleBinding := bootstrapRoleBindings[i]
			names.Insert(roleBinding.Name)
			roleBindings[roleBinding.Name] = &roleBinding
		}

		for _, name := range names.List() {
			list.Items = append(list.Items, roleBindings[name])
		}
	}

	testObjects(t, list, "namespace-role-bindings.yaml")
}

func TestBootstrapClusterRoles(t *testing.T) {
	list := &api.List{}
	names := sets.NewString()
	roles := map[string]runtime.Object{}
	bootstrapRoles := bootstrappolicy.ClusterRoles()
	for i := range bootstrapRoles {
		role := bootstrapRoles[i]
		names.Insert(role.Name)
		roles[role.Name] = &role
	}
	for _, name := range names.List() {
		list.Items = append(list.Items, roles[name])
	}
	testObjects(t, list, "cluster-roles.yaml")
}

func TestBootstrapClusterRoleBindings(t *testing.T) {
	list := &api.List{}
	names := sets.NewString()
	roleBindings := map[string]runtime.Object{}
	bootstrapRoleBindings := bootstrappolicy.ClusterRoleBindings()
	for i := range bootstrapRoleBindings {
		role := bootstrapRoleBindings[i]
		names.Insert(role.Name)
		roleBindings[role.Name] = &role
	}
	for _, name := range names.List() {
		list.Items = append(list.Items, roleBindings[name])
	}
	testObjects(t, list, "cluster-role-bindings.yaml")
}

func TestBootstrapControllerRoles(t *testing.T) {
	list := &api.List{}
	names := sets.NewString()
	roles := map[string]runtime.Object{}
	bootstrapRoles := bootstrappolicy.ControllerRoles()
	for i := range bootstrapRoles {
		role := bootstrapRoles[i]
		names.Insert(role.Name)
		roles[role.Name] = &role
	}
	for _, name := range names.List() {
		list.Items = append(list.Items, roles[name])
	}
	testObjects(t, list, "controller-roles.yaml")
}

func TestBootstrapControllerRoleBindings(t *testing.T) {
	list := &api.List{}
	names := sets.NewString()
	roleBindings := map[string]runtime.Object{}
	bootstrapRoleBindings := bootstrappolicy.ControllerRoleBindings()
	for i := range bootstrapRoleBindings {
		roleBinding := bootstrapRoleBindings[i]
		names.Insert(roleBinding.Name)
		roleBindings[roleBinding.Name] = &roleBinding
	}
	for _, name := range names.List() {
		list.Items = append(list.Items, roleBindings[name])
	}
	testObjects(t, list, "controller-role-bindings.yaml")
}

func testObjects(t *testing.T, list *api.List, fixtureFilename string) {
	filename := filepath.Join("testdata", fixtureFilename)
	expectedYAML, err := os.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}

	if err := runtime.EncodeList(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion, rbacv1.SchemeGroupVersion), list.Items); err != nil {
		t.Fatal(err)
	}

	jsonData, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion, rbacv1.SchemeGroupVersion), list)
	if err != nil {
		t.Fatal(err)
	}
	yamlData, err := yaml.JSONToYAML(jsonData)
	if err != nil {
		t.Fatal(err)
	}
	if string(yamlData) != string(expectedYAML) {
		t.Errorf("Bootstrap policy data does not match the test fixture in %s", filename)

		const updateEnvVar = "UPDATE_BOOTSTRAP_POLICY_FIXTURE_DATA"
		if os.Getenv(updateEnvVar) == "true" {
			if err := os.WriteFile(filename, []byte(yamlData), os.FileMode(0755)); err == nil {
				t.Logf("Updated data in %s", filename)
				t.Logf("Verify the diff, commit changes, and rerun the tests")
			} else {
				t.Logf("Could not update data in %s: %v", filename, err)
			}
		} else {
			t.Logf("Diff between bootstrap data and fixture data in %s:\n-------------\n%s", filename, cmp.Diff(string(yamlData), string(expectedYAML)))
			t.Logf("If the change is expected, re-run with %s=true to update the fixtures", updateEnvVar)
		}
	}
}

func TestClusterRoleLabel(t *testing.T) {
	roles := bootstrappolicy.ClusterRoles()
	for i := range roles {
		role := roles[i]
		accessor, err := meta.Accessor(&role)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if accessor.GetLabels()["kubernetes.io/bootstrapping"] != "rbac-defaults" {
			t.Errorf("ClusterRole: %s GetLabels() = %s, want %s", accessor.GetName(), accessor.GetLabels(), map[string]string{"kubernetes.io/bootstrapping": "rbac-defaults"})
		}
	}

	rolebindings := bootstrappolicy.ClusterRoleBindings()
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
