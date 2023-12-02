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

	"k8s.io/klog/v2"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

var (
	// namespaceRoles is a map of namespace to slice of roles to create
	namespaceRoles = map[string][]rbacv1.Role{}

	// namespaceRoleBindings is a map of namespace to slice of roleBindings to create
	namespaceRoleBindings = map[string][]rbacv1.RoleBinding{}
)

func addNamespaceRole(namespace string, role rbacv1.Role) {
	if !strings.HasPrefix(namespace, "kube-") {
		klog.Fatalf(`roles can only be bootstrapped into reserved namespaces starting with "kube-", not %q`, namespace)
	}

	existingRoles := namespaceRoles[namespace]
	for _, existingRole := range existingRoles {
		if role.Name == existingRole.Name {
			klog.Fatalf("role %q was already registered in %q", role.Name, namespace)
		}
	}

	role.Namespace = namespace
	addDefaultMetadata(&role)
	existingRoles = append(existingRoles, role)
	namespaceRoles[namespace] = existingRoles
}

func addNamespaceRoleBinding(namespace string, roleBinding rbacv1.RoleBinding) {
	if !strings.HasPrefix(namespace, "kube-") {
		klog.Fatalf(`rolebindings can only be bootstrapped into reserved namespaces starting with "kube-", not %q`, namespace)
	}

	existingRoleBindings := namespaceRoleBindings[namespace]
	for _, existingRoleBinding := range existingRoleBindings {
		if roleBinding.Name == existingRoleBinding.Name {
			klog.Fatalf("rolebinding %q was already registered in %q", roleBinding.Name, namespace)
		}
	}

	roleBinding.Namespace = namespace
	addDefaultMetadata(&roleBinding)
	existingRoleBindings = append(existingRoleBindings, roleBinding)
	namespaceRoleBindings[namespace] = existingRoleBindings
}

func init() {
	addNamespaceRole(metav1.NamespaceSystem, rbacv1.Role{
		// role for finding authentication config info for starting a server
		ObjectMeta: metav1.ObjectMeta{Name: "extension-apiserver-authentication-reader"},
		Rules: []rbacv1.PolicyRule{
			// this particular config map is exposed and contains authentication configuration information
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("configmaps").Names("extension-apiserver-authentication").RuleOrDie(),
		},
	})
	addNamespaceRole(metav1.NamespaceSystem, rbacv1.Role{
		// role for the bootstrap signer to be able to inspect kube-system secrets
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "bootstrap-signer"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
		},
	})
	addNamespaceRole(metav1.NamespaceSystem, rbacv1.Role{
		// role for the cloud providers to access/create kube-system configmaps
		// Deprecated starting Kubernetes 1.10 and will be deleted according to GA deprecation policy.
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "cloud-provider"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("create", "get", "list", "watch").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
		},
	})
	addNamespaceRole(metav1.NamespaceSystem, rbacv1.Role{
		// role for the token-cleaner to be able to remove secrets, but only in kube-system
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "token-cleaner"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "delete").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
			eventsRule(),
		},
	})
	// TODO: Create util on Role+Binding for leader locking if more cases evolve.
	addNamespaceRole(metav1.NamespaceSystem, rbacv1.Role{
		// role for the leader locking on supplied configmap
		ObjectMeta: metav1.ObjectMeta{Name: "system::leader-locking-kube-controller-manager"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("watch").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
			rbacv1helpers.NewRule("get", "update").Groups(legacyGroup).Resources("configmaps").Names("kube-controller-manager").RuleOrDie(),
			rbacv1helpers.NewRule("get", "watch", "list", "create", "update").Groups("coordination.k8s.io").Resources("leases").RuleOrDie(),
		},
	})
	addNamespaceRole(metav1.NamespaceSystem, rbacv1.Role{
		// role for the leader locking on supplied configmap
		ObjectMeta: metav1.ObjectMeta{Name: "system::leader-locking-kube-scheduler"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("watch").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
			rbacv1helpers.NewRule("get", "update").Groups(legacyGroup).Resources("configmaps").Names("kube-scheduler").RuleOrDie(),
		},
	})

	delegatedAuthBinding := rbacv1helpers.NewRoleBinding("extension-apiserver-authentication-reader", metav1.NamespaceSystem).Users(user.KubeControllerManager, user.KubeScheduler).BindingOrDie()
	delegatedAuthBinding.Name = "system::extension-apiserver-authentication-reader"
	addNamespaceRoleBinding(metav1.NamespaceSystem, delegatedAuthBinding)

	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbacv1helpers.NewRoleBinding("system::leader-locking-kube-controller-manager", metav1.NamespaceSystem).Users(user.KubeControllerManager).SAs(metav1.NamespaceSystem, "kube-controller-manager").BindingOrDie())
	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbacv1helpers.NewRoleBinding("system::leader-locking-kube-scheduler", metav1.NamespaceSystem).Users(user.KubeScheduler).SAs(metav1.NamespaceSystem, "kube-scheduler").BindingOrDie())
	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbacv1helpers.NewRoleBinding(saRolePrefix+"bootstrap-signer", metav1.NamespaceSystem).SAs(metav1.NamespaceSystem, "bootstrap-signer").BindingOrDie())
	// cloud-provider is deprecated starting Kubernetes 1.10 and will be deleted according to GA deprecation policy.
	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbacv1helpers.NewRoleBinding(saRolePrefix+"cloud-provider", metav1.NamespaceSystem).SAs(metav1.NamespaceSystem, "cloud-provider").BindingOrDie())
	addNamespaceRoleBinding(metav1.NamespaceSystem,
		rbacv1helpers.NewRoleBinding(saRolePrefix+"token-cleaner", metav1.NamespaceSystem).SAs(metav1.NamespaceSystem, "token-cleaner").BindingOrDie())

	addNamespaceRole(metav1.NamespacePublic, rbacv1.Role{
		// role for the bootstrap signer to be able to write its configmap
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "bootstrap-signer"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("configmaps").Names("cluster-info").RuleOrDie(),
			eventsRule(),
		},
	})
	addNamespaceRoleBinding(metav1.NamespacePublic,
		rbacv1helpers.NewRoleBinding(saRolePrefix+"bootstrap-signer", metav1.NamespacePublic).SAs(metav1.NamespaceSystem, "bootstrap-signer").BindingOrDie())

}

// NamespaceRoles returns a map of namespace to slice of roles to create
func NamespaceRoles() map[string][]rbacv1.Role {
	return namespaceRoles
}

// NamespaceRoleBindings returns a map of namespace to slice of roles to create
func NamespaceRoleBindings() map[string][]rbacv1.RoleBinding {
	return namespaceRoleBindings
}
