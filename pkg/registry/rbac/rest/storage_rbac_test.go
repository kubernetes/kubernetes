/*
Copyright 2021 The Kubernetes Authors.

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

package rest

import (
	"testing"

	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

func BenchmarkEnsureRBACPolicy(b *testing.B) {
	for n := 0; n < b.N; n++ {
		var policy = &PolicyData{
			ClusterRoles:               append(bootstrappolicy.ClusterRoles(), bootstrappolicy.ControllerRoles()...),
			ClusterRoleBindings:        append(bootstrappolicy.ClusterRoleBindings(), bootstrappolicy.ControllerRoleBindings()...),
			Roles:                      bootstrappolicy.NamespaceRoles(),
			RoleBindings:               bootstrappolicy.NamespaceRoleBindings(),
			ClusterRolesToAggregate:    bootstrappolicy.ClusterRolesToAggregate(),
			ClusterRoleBindingsToSplit: bootstrappolicy.ClusterRoleBindingsToSplit(),
		}
		coreClientSet := fake.NewSimpleClientset()
		_, _ = ensureRBACPolicy(policy, coreClientSet)
	}
}
