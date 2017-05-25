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
	"strings"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
)

var (
	// namespaceRoles is a map of namespace to slice of roles to create
	namespaceRoles = map[string][]rbac.Role{}

	// namespaceRoleBindings is a map of namespace to slice of roleBindings to create
	namespaceRoleBindings = map[string][]rbac.RoleBinding{}
)

func addNamespaceRole(namespace string, role rbac.Role) {
	if !strings.HasPrefix(namespace, "kube-") {
		glog.Fatalf(`roles can only be bootstrapped into reserved namespaces starting with "kube-", not %q`, namespace)
	}

	existingRoles := namespaceRoles[namespace]
	for _, existingRole := range existingRoles {
		if role.Name == existingRole.Name {
			glog.Fatalf("role %q was already registered in %q", role.Name, namespace)
		}
	}

	role.Namespace = namespace
	addDefaultMetadata(&role)
	existingRoles = append(existingRoles, role)
	namespaceRoles[namespace] = existingRoles
}

func addNamespaceRoleBinding(namespace string, roleBinding rbac.RoleBinding) {
	if !strings.HasPrefix(namespace, "kube-") {
		glog.Fatalf(`roles can only be bootstrapped into reserved namespaces starting with "kube-", not %q`, namespace)
	}

	existingRoleBindings := namespaceRoleBindings[namespace]
	for _, existingRoleBinding := range existingRoleBindings {
		if roleBinding.Name == existingRoleBinding.Name {
			glog.Fatalf("rolebinding %q was already registered in %q", roleBinding.Name, namespace)
		}
	}

	roleBinding.Namespace = namespace
	addDefaultMetadata(&roleBinding)
	existingRoleBindings = append(existingRoleBindings, roleBinding)
	namespaceRoleBindings[namespace] = existingRoleBindings
}

func init() {
	addNamespaceRole(metav1.NamespaceSystem, rbac.Role{
		// role for finding authentication config info for starting a server
		ObjectMeta: metav1.ObjectMeta{Name: "extension-apiserver-authentication-reader"},
		Rules: []rbac.PolicyRule{
			// this particular config map is exposed and contains authentication configuration information
			rbac.NewRule("get").Groups(legacyGroup).Resources("configmaps").Names("extension-apiserver-authentication").RuleOrDie(),
		},
	})
	addNamespaceRole(metav1.NamespaceSystem, rbac.Role{
		// role for the bootstrap signer to be able to inspect kube-system secrets
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "bootstrap-signer"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
		},
	})
	addNamespaceRole(metav1.NamespaceSystem, rbac.Role{
		// role for the cloud providers to access/create kube-system configmaps
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "cloud-provider"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("create", "get", "list", "watch").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
		},
	})
	addNamespaceRole(metav1.NamespaceSystem, rbac.Role{
		// role for the token-cleaner to be able to remove secrets, but only in kube-system
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "token-cleaner"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch", "delete").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
			eventsRule(),
		},
	})
	// TODO: Create util on Role+Binding for leader locking if more cases evolve.
	addNamespaceRole(metav1.NamespaceSystem, rbac.Role{
		// role for the leader locking on supplied configmap
		ObjectMeta: metav1.ObjectMeta{Name: "system::leader-locking-kube-controller-manager"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("watch").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
			rbac.NewRule("get", "update").Groups(legacyGroup).Resources("configmaps").Names("kube-controller-manager").RuleOrDie(),
		},
	})
	addNamespaceRole(metav1.NamespaceSystem, rbac.Role{
		// role for the leader locking on supplied configmap
		ObjectMeta: metav1.ObjectMeta{Name: "system::leader-locking-kube-scheduler"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("watch").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
			rbac.NewRule("get", "update").Groups(legacyGroup).Resources("configmaps").Names("kube-scheduler").RuleOrDie(),
		},
	})
	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbac.NewRoleBinding("system::leader-locking-kube-controller-manager", metav1.NamespaceSystem).SAs(metav1.NamespaceSystem, "kube-controller-manager").BindingOrDie())
	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbac.NewRoleBinding("system::leader-locking-kube-scheduler", metav1.NamespaceSystem).SAs(metav1.NamespaceSystem, "kube-scheduler").BindingOrDie())
	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbac.NewRoleBinding(saRolePrefix+"bootstrap-signer", metav1.NamespaceSystem).SAs(metav1.NamespaceSystem, "bootstrap-signer").BindingOrDie())
	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbac.NewRoleBinding(saRolePrefix+"cloud-provider", metav1.NamespaceSystem).SAs(metav1.NamespaceSystem, "cloud-provider").BindingOrDie())
	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbac.NewRoleBinding(saRolePrefix+"token-cleaner", metav1.NamespaceSystem).SAs(metav1.NamespaceSystem, "token-cleaner").BindingOrDie())

	addNamespaceRole(metav1.NamespacePublic, rbac.Role{
		// role for the bootstrap signer to be able to write its configmap
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "bootstrap-signer"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
			rbac.NewRule("update").Groups(legacyGroup).Resources("configmaps").Names("cluster-info").RuleOrDie(),
			eventsRule(),
		},
	})
	addNamespaceRoleBinding(metav1.NamespacePublic,
		rbac.NewRoleBinding(saRolePrefix+"bootstrap-signer", metav1.NamespacePublic).SAs(metav1.NamespaceSystem, "bootstrap-signer").BindingOrDie())

}

// NamespaceRoles returns a map of namespace to slice of roles to create
func NamespaceRoles() map[string][]rbac.Role {
	return namespaceRoles
}

// NamespaceRoleBindings returns a map of namespace to slice of roles to create
func NamespaceRoleBindings() map[string][]rbac.RoleBinding {
	return namespaceRoleBindings
}
