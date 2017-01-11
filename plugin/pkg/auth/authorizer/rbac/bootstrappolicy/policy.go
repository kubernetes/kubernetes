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
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
)

var (
	ReadWrite = []string{"get", "list", "watch", "create", "update", "patch", "delete", "deletecollection"}
	Read      = []string{"get", "list", "watch"}

	Label = map[string]string{"kubernetes.io/bootstrapping": "rbac-defaults"}
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

func addClusterRoleLabel(roles []rbac.ClusterRole) {
	for i := range roles {
		if roles[i].ObjectMeta.Labels == nil {
			roles[i].ObjectMeta.Labels = make(map[string]string)
		}
		for k, v := range Label {
			roles[i].ObjectMeta.Labels[k] = v
		}
	}
	return
}

func addClusterRoleBindingLabel(rolebindings []rbac.ClusterRoleBinding) {
	for i := range rolebindings {
		if rolebindings[i].ObjectMeta.Labels == nil {
			rolebindings[i].ObjectMeta.Labels = make(map[string]string)
		}
		for k, v := range Label {
			rolebindings[i].ObjectMeta.Labels[k] = v
		}
	}
	return
}

// ClusterRoles returns the cluster roles to bootstrap an API server with
func ClusterRoles() []rbac.ClusterRole {
	roles := []rbac.ClusterRole{
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
				rbac.NewRule("get").URLs("/version", "/swaggerapi", "/swaggerapi/*", "/api", "/api/*", "/apis", "/apis/*").RuleOrDie(),
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

				rbac.NewRule(ReadWrite...).Groups(batchGroup).Resources("jobs", "cronjobs", "scheduledjobs").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(extensionsGroup).Resources("daemonsets", "horizontalpodautoscalers",
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

				rbac.NewRule(ReadWrite...).Groups(batchGroup).Resources("jobs", "cronjobs", "scheduledjobs").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(extensionsGroup).Resources("daemonsets", "horizontalpodautoscalers",
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

				rbac.NewRule(Read...).Groups(batchGroup).Resources("jobs", "cronjobs", "scheduledjobs").RuleOrDie(),

				rbac.NewRule(Read...).Groups(extensionsGroup).Resources("daemonsets", "horizontalpodautoscalers",
					"replicationcontrollers/scale", "replicasets", "replicasets/scale", "deployments", "deployments/scale").RuleOrDie(),
			},
		},
		{
			// a role for nodes to use to have the access they need for running pods
			ObjectMeta: api.ObjectMeta{Name: "system:node"},
			Rules: []rbac.PolicyRule{
				// Needed to check API access.  These creates are non-mutating
				rbac.NewRule("create").Groups(authenticationGroup).Resources("tokenreviews").RuleOrDie(),
				rbac.NewRule("create").Groups(authorizationGroup).Resources("subjectaccessreviews", "localsubjectaccessreviews").RuleOrDie(),
				// Needed to build serviceLister, to populate env vars for services
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("services").RuleOrDie(),
				// Nodes can register themselves
				// TODO: restrict to creating a node with the same name they announce
				rbac.NewRule("create", "get", "list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
				// TODO: restrict to the bound node once supported
				rbac.NewRule("update", "patch").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),

				// TODO: restrict to the bound node as creator once supported
				rbac.NewRule("create", "update", "patch").Groups(legacyGroup).Resources("events").RuleOrDie(),

				// TODO: restrict to pods scheduled on the bound node once supported
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("pods").RuleOrDie(),

				// TODO: remove once mirror pods are removed
				// TODO: restrict deletion to mirror pods created by the bound node once supported
				// Needed for the node to create/delete mirror pods
				rbac.NewRule("get", "create", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
				// TODO: restrict to pods scheduled on the bound node once supported
				rbac.NewRule("update").Groups(legacyGroup).Resources("pods/status").RuleOrDie(),

				// TODO: restrict to secrets and configmaps used by pods scheduled on bound node once supported
				// Needed for imagepullsecrets, rbd/ceph and secret volumes, and secrets in envs
				// Needed for configmap volume and envs
				rbac.NewRule("get").Groups(legacyGroup).Resources("secrets", "configmaps").RuleOrDie(),
				// TODO: restrict to claims/volumes used by pods scheduled on bound node once supported
				// Needed for persistent volumes
				rbac.NewRule("get").Groups(legacyGroup).Resources("persistentvolumeclaims", "persistentvolumes").RuleOrDie(),
				// TODO: restrict to namespaces of pods scheduled on bound node once supported
				// TODO: change glusterfs to use DNS lookup so this isn't needed?
				// Needed for glusterfs volumes
				rbac.NewRule("get").Groups(legacyGroup).Resources("endpoints").RuleOrDie(),
			},
		},
		{
			// a role to use for setting up a proxy
			ObjectMeta: api.ObjectMeta{Name: "system:node-proxier"},
			Rules: []rbac.PolicyRule{
				// Used to build serviceLister
				rbac.NewRule("list", "watch").Groups(legacyGroup).Resources("services", "endpoints").RuleOrDie(),
				rbac.NewRule("get").Groups(legacyGroup).Resources("nodes").RuleOrDie(),

				eventsRule(),
			},
		},
		{
			// a role to use for allowing authentication and authorization delegation
			ObjectMeta: api.ObjectMeta{Name: "system:auth-delegator"},
			Rules: []rbac.PolicyRule{
				// These creates are non-mutating
				rbac.NewRule("create").Groups(authenticationGroup).Resources("tokenreviews").RuleOrDie(),
				rbac.NewRule("create").Groups(authorizationGroup).Resources("subjectaccessreviews").RuleOrDie(),
			},
		},
		{
			// a role to use for the API registry, summarization, and proxy handling
			ObjectMeta: api.ObjectMeta{Name: "system:kube-aggregator"},
			Rules: []rbac.PolicyRule{
				// it needs to see all services so that it knows whether the ones it points to exist or not
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("services", "endpoints").RuleOrDie(),
			},
		},
		{
			// a role to use for bootstrapping the kube-controller-manager so it can create the shared informers
			// service accounts, and secrets that we need to create separate identities for other controllers
			ObjectMeta: api.ObjectMeta{Name: "system:kube-controller-manager"},
			Rules: []rbac.PolicyRule{
				eventsRule(),
				rbac.NewRule("create").Groups(legacyGroup).Resources("endpoints", "secrets", "serviceaccounts").RuleOrDie(),
				rbac.NewRule("delete").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
				rbac.NewRule("get").Groups(legacyGroup).Resources("endpoints", "namespaces", "serviceaccounts").RuleOrDie(),
				rbac.NewRule("update").Groups(legacyGroup).Resources("endpoints", "serviceaccounts").RuleOrDie(),

				rbac.NewRule("list", "watch").Groups("*").Resources("namespaces", "nodes", "persistentvolumeclaims",
					"persistentvolumes", "pods", "secrets", "serviceaccounts", "replicationcontrollers").RuleOrDie(),
				rbac.NewRule("list", "watch").Groups(extensionsGroup).Resources("daemonsets", "deployments", "replicasets").RuleOrDie(),
				rbac.NewRule("list", "watch").Groups(batchGroup).Resources("jobs", "cronjobs").RuleOrDie(),
			},
		},
	}
	addClusterRoleLabel(roles)
	return roles
}

// ClusterRoleBindings return default rolebindings to the default roles
func ClusterRoleBindings() []rbac.ClusterRoleBinding {
	rolebindings := []rbac.ClusterRoleBinding{
		rbac.NewClusterBinding("cluster-admin").Groups(user.SystemPrivilegedGroup).BindingOrDie(),
		rbac.NewClusterBinding("system:discovery").Groups(user.AllAuthenticated, user.AllUnauthenticated).BindingOrDie(),
		rbac.NewClusterBinding("system:basic-user").Groups(user.AllAuthenticated, user.AllUnauthenticated).BindingOrDie(),
		rbac.NewClusterBinding("system:node").Groups(user.NodesGroup).BindingOrDie(),
		rbac.NewClusterBinding("system:node-proxier").Users(user.KubeProxy).BindingOrDie(),
		rbac.NewClusterBinding("system:kube-controller-manager").Users(user.KubeControllerManager).BindingOrDie(),
	}
	addClusterRoleBindingLabel(rolebindings)
	return rolebindings
}
