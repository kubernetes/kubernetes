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
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac"
)

// ClusterRoles returns the cluster roles to bootstrap an API server with
func ClusterRoles() []rbacapi.ClusterRole {
	return []rbacapi.ClusterRole{
		// TODO update the expression of these rules to match openshift for ease of inspection
		{
			ObjectMeta: api.ObjectMeta{Name: "cluster-admin"},
			Rules: []rbacapi.PolicyRule{
				{Verbs: []string{"*"}, APIGroups: []string{"*"}, Resources: []string{"*"}},
				{Verbs: []string{"*"}, NonResourceURLs: []string{"*"}},
			},
		},
	}
}
