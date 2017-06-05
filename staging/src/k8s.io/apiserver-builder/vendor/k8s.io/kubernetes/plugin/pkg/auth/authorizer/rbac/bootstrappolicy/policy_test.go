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
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/ghodss/yaml"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	rbacv1beta1 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	rbacregistryvalidation "k8s.io/kubernetes/pkg/registry/rbac/validation"
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

	if covers, miss := rbacregistryvalidation.Covers(semanticRoles.admin.Rules, semanticRoles.edit.Rules); !covers {
		t.Errorf("failed to cover: %#v", miss)
	}
	if covers, miss := rbacregistryvalidation.Covers(semanticRoles.admin.Rules, semanticRoles.view.Rules); !covers {
		t.Errorf("failed to cover: %#v", miss)
	}
	if covers, miss := rbacregistryvalidation.Covers(semanticRoles.edit.Rules, semanticRoles.view.Rules); !covers {
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
		if covers, _ := rbacregistryvalidation.Covers(semanticRoles.edit.Rules, []rbac.PolicyRule{rule}); covers {
			t.Errorf("edit has extra powers: %#v", rule)
		}
	}
	semanticRoles.edit.Rules = append(semanticRoles.edit.Rules, additionalAdminPowers...)

	// at this point, we should have a two way covers relationship
	if covers, miss := rbacregistryvalidation.Covers(semanticRoles.admin.Rules, semanticRoles.edit.Rules); !covers {
		t.Errorf("admin has lost rules for: %#v", miss)
	}
	if covers, miss := rbacregistryvalidation.Covers(semanticRoles.edit.Rules, semanticRoles.admin.Rules); !covers {
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

// ungettableResources is the list of rules that don't allow to view (GET) them
// this is purposefully separate list to distinguish from escalating privs
var ungettableResources = []rbac.PolicyRule{
	rbac.NewRule(bootstrappolicy.Read...).Groups("apps", "extensions").Resources("deployments/rollback").RuleOrDie(),
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
		if covers, _ := rbacregistryvalidation.Covers(semanticRoles.view.Rules, []rbac.PolicyRule{rule}); covers {
			t.Errorf("view has extra powers: %#v", rule)
		}
	}
	semanticRoles.view.Rules = append(semanticRoles.view.Rules, viewEscalatingNamespaceResources...)

	// confirm that the view role doesn't have ungettable resources
	for _, rule := range ungettableResources {
		if covers, _ := rbacregistryvalidation.Covers(semanticRoles.view.Rules, []rbac.PolicyRule{rule}); covers {
			t.Errorf("view has ungettable resource: %#v", rule)
		}
	}
	semanticRoles.view.Rules = append(semanticRoles.view.Rules, ungettableResources...)

	// at this point, we should have a two way covers relationship
	if covers, miss := rbacregistryvalidation.Covers(semanticRoles.edit.Rules, semanticRoles.view.Rules); !covers {
		t.Errorf("edit has lost rules for: %#v", miss)
	}
	if covers, miss := rbacregistryvalidation.Covers(semanticRoles.view.Rules, semanticRoles.edit.Rules); !covers {
		t.Errorf("view is missing rules for: %#v\nIf these are escalating powers, add them to the list.  Otherwise, add them to the view role.", miss)
	}
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
	expectedYAML, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}

	if err := runtime.EncodeList(api.Codecs.LegacyCodec(v1.SchemeGroupVersion, rbacv1beta1.SchemeGroupVersion), list.Items); err != nil {
		t.Fatal(err)
	}

	jsonData, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion, rbacv1beta1.SchemeGroupVersion), list)
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
			if err := ioutil.WriteFile(filename, []byte(yamlData), os.FileMode(0755)); err == nil {
				t.Logf("Updated data in %s", filename)
				t.Logf("Verify the diff, commit changes, and rerun the tests")
			} else {
				t.Logf("Could not update data in %s: %v", filename, err)
			}
		} else {
			t.Logf("Diff between bootstrap data and fixture data in %s:\n-------------\n%s", filename, diff.StringDiff(string(yamlData), string(expectedYAML)))
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
		if got, want := accessor.GetLabels(), map[string]string{"kubernetes.io/bootstrapping": "rbac-defaults"}; !reflect.DeepEqual(got, want) {
			t.Errorf("ClusterRole: %s GetLabels() = %s, want %s", accessor.GetName(), got, want)
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
