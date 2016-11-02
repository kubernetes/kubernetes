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
	"k8s.io/kubernetes/pkg/auth/user"
)

var (
	ReadWrite = []string{"get", "list", "watch", "create", "update", "patch", "delete", "deletecollection"}
	Read      = []string{"get", "list", "watch"}
)

const (
	legacyGroup         = ""
	appsGroup           = "apps"
	authenticationGroup = "authentication.k8s.io"
	authorizationGroup  = "authorization.k8s.io"
	autoscalingGroup    = "autoscaling"
	batchGroup          = "batch"
	certificatesGroup   = "certificates.k8s.io"
	extensionsGroup     = "extensions"
	policyGroup         = "policy"
	rbacGroup           = "rbac.authorization.k8s.io"
	storageGroup        = "storage.k8s.io"
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
		{
			// a role which provides minimal resource access to allow a "normal" user to learn information about themselves
			ObjectMeta: api.ObjectMeta{Name: "system:basic-user"},
			Rules: []rbac.PolicyRule{
				// TODO add future selfsubjectrulesreview, project request APIs, project listing APIs
				rbac.NewRule("create").Groups(authorizationGroup).Resources("selfsubjectaccessreviews").RuleOrDie(),
			},
		},

		{
			// a role for a namespace level admin.  It is `edit` plus the power to grant permissions to other users.
			ObjectMeta: api.ObjectMeta{Name: "admin"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule(ReadWrite...).Groups(legacyGroup).Resources("pods", "pods/attach", "pods/proxy", "pods/exec", "pods/portforward").RuleOrDie(),
				rbac.NewRule(ReadWrite...).Groups(legacyGroup).Resources("replicationcontrollers", "replicationcontrollers/scale", "serviceaccounts",
					"services", "services/proxy", "endpoints", "persistentvolumeclaims", "configmaps", "secrets").RuleOrDie(),
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("limitranges", "resourcequotas", "bindings", "events",
					"pods/status", "resourcequotas/status", "namespaces/status", "replicationcontrollers/status", "pods/log").RuleOrDie(),
				// read access to namespaces at the namespace scope means you can read *this* namespace.  This can be used as an
				// indicator of which namespaces you have access to.
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("namespaces").RuleOrDie(),
				rbac.NewRule("impersonate").Groups(legacyGroup).Resources("serviceaccounts").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(appsGroup).Resources("statefulsets").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(autoscalingGroup).Resources("horizontalpodautoscalers").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(batchGroup).Resources("jobs", "scheduledjobs").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(extensionsGroup).Resources("jobs", "daemonsets", "horizontalpodautoscalers",
					"replicationcontrollers/scale", "replicasets", "replicasets/scale", "deployments", "deployments/scale").RuleOrDie(),

				// additional admin powers
				rbac.NewRule("create").Groups(authorizationGroup).Resources("localsubjectaccessreviews").RuleOrDie(),
				rbac.NewRule(ReadWrite...).Groups(rbacGroup).Resources("roles", "rolebindings").RuleOrDie(),
			},
		},
		{
			// a role for a namespace level editor.  It grants access to all user level actions in a namespace.
			// It does not grant powers for "privileged" resources which are domain of the system: `/status`
			// subresources or `quota`/`limits` which are used to control namespaces
			ObjectMeta: api.ObjectMeta{Name: "edit"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule(ReadWrite...).Groups(legacyGroup).Resources("pods", "pods/attach", "pods/proxy", "pods/exec", "pods/portforward").RuleOrDie(),
				rbac.NewRule(ReadWrite...).Groups(legacyGroup).Resources("replicationcontrollers", "replicationcontrollers/scale", "serviceaccounts",
					"services", "services/proxy", "endpoints", "persistentvolumeclaims", "configmaps", "secrets").RuleOrDie(),
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("limitranges", "resourcequotas", "bindings", "events",
					"pods/status", "resourcequotas/status", "namespaces/status", "replicationcontrollers/status", "pods/log").RuleOrDie(),
				// read access to namespaces at the namespace scope means you can read *this* namespace.  This can be used as an
				// indicator of which namespaces you have access to.
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("namespaces").RuleOrDie(),
				rbac.NewRule("impersonate").Groups(legacyGroup).Resources("serviceaccounts").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(appsGroup).Resources("statefulsets").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(autoscalingGroup).Resources("horizontalpodautoscalers").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(batchGroup).Resources("jobs", "scheduledjobs").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(extensionsGroup).Resources("jobs", "daemonsets", "horizontalpodautoscalers",
					"replicationcontrollers/scale", "replicasets", "replicasets/scale", "deployments", "deployments/scale").RuleOrDie(),
			},
		},
		{
			// a role for namespace level viewing.  It grants Read-only access to non-escalating resources in
			// a namespace.
			ObjectMeta: api.ObjectMeta{Name: "view"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("pods", "replicationcontrollers", "replicationcontrollers/scale", "serviceaccounts",
					"services", "endpoints", "persistentvolumeclaims", "configmaps").RuleOrDie(),
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("limitranges", "resourcequotas", "bindings", "events",
					"pods/status", "resourcequotas/status", "namespaces/status", "replicationcontrollers/status", "pods/log").RuleOrDie(),
				// read access to namespaces at the namespace scope means you can read *this* namespace.  This can be used as an
				// indicator of which namespaces you have access to.
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("namespaces").RuleOrDie(),

				rbac.NewRule(Read...).Groups(appsGroup).Resources("statefulsets").RuleOrDie(),

				rbac.NewRule(Read...).Groups(autoscalingGroup).Resources("horizontalpodautoscalers").RuleOrDie(),

				rbac.NewRule(Read...).Groups(batchGroup).Resources("jobs", "scheduledjobs").RuleOrDie(),

				rbac.NewRule(Read...).Groups(extensionsGroup).Resources("jobs", "daemonsets", "horizontalpodautoscalers",
					"replicationcontrollers/scale", "replicasets", "replicasets/scale", "deployments", "deployments/scale").RuleOrDie(),
			},
		},
	}
}

// ClusterRoleBindings return default rolebindings to the default roles
func ClusterRoleBindings() []rbac.ClusterRoleBinding {
	return []rbac.ClusterRoleBinding{
		rbac.NewClusterBinding("cluster-admin").Groups(user.SystemPrivilegedGroup).BindingOrDie(),
		rbac.NewClusterBinding("system:discovery").Groups(user.AllAuthenticated, user.AllUnauthenticated).BindingOrDie(),
		rbac.NewClusterBinding("system:basic-user").Groups(user.AllAuthenticated, user.AllUnauthenticated).BindingOrDie(),
	}
}
