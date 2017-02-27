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

func init() {
	addNamespaceRole(metav1.NamespaceSystem, rbac.Role{
		// role for finding authentication config info for starting a server
		ObjectMeta: metav1.ObjectMeta{Name: "extension-apiserver-authentication-reader"},
		Rules: []rbac.PolicyRule{
			// this particular config map is exposed and contains authentication configuration information
			rbac.NewRule("get").Groups(legacyGroup).Resources("configmaps").Names("extension-apiserver-authentication").RuleOrDie(),
		},
	})
}

// NamespaceRoles returns a map of namespace to slice of roles to create
func NamespaceRoles() map[string][]rbac.Role {
	return namespaceRoles
}
