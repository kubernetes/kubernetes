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
	"k8s.io/kubernetes/pkg/api"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
)

var (
	readWrite = []string{"get", "list", "watch", "create", "update", "patch", "delete", "deletecollection"}
	read      = []string{"get", "list", "watch"}

	legacyGroup = ""
)

// ClusterRoles returns the cluster roles to bootstrap an API server with
func ClusterRoles() []rbac.ClusterRole {
	return []rbac.ClusterRole{
		{
			// a "root" role which can do absolutely anything
			ObjectMeta: api.ObjectMeta{Name: "cluster-admin"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("*").Groups("*").Resources("*").RuleOrDie(),
				rbac.NewRule("*").URLs("*").RuleOrDie(),
			},
		},
		{
			// a role which provides just enough power to discovery API versions for negotiation
			ObjectMeta: api.ObjectMeta{Name: "system:discovery"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("get").URLs("/version", "/api", "/api/*", "/apis", "/apis/*").RuleOrDie(),
			},
		},
	}
}
