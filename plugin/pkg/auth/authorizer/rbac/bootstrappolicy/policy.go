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
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/features"
)

var (
	ReadWrite = []string{"get", "list", "watch", "create", "update", "patch", "delete", "deletecollection"}
	Read      = []string{"get", "list", "watch"}

	Label      = map[string]string{"kubernetes.io/bootstrapping": "rbac-defaults"}
	Annotation = map[string]string{rbac.AutoUpdateAnnotationKey: "true"}
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
	resMetricsGroup     = "metrics.k8s.io"
	customMetricsGroup  = "custom.metrics.k8s.io"
)

func addDefaultMetadata(obj runtime.Object) {
	metadata, err := meta.Accessor(obj)
	if err != nil {
		// if this happens, then some static code is broken
		panic(err)
	}

	labels := metadata.GetLabels()
	if labels == nil {
		labels = map[string]string{}
	}
	for k, v := range Label {
		labels[k] = v
	}
	metadata.SetLabels(labels)

	annotations := metadata.GetAnnotations()
	if annotations == nil {
		annotations = map[string]string{}
	}
	for k, v := range Annotation {
		annotations[k] = v
	}
	metadata.SetAnnotations(annotations)
}

func addClusterRoleLabel(roles []rbac.ClusterRole) {
	for i := range roles {
		addDefaultMetadata(&roles[i])
	}
	return
}

func addClusterRoleBindingLabel(rolebindings []rbac.ClusterRoleBinding) {
	for i := range rolebindings {
		addDefaultMetadata(&rolebindings[i])
	}
	return
}

func NodeRules() []rbac.PolicyRule {
	return []rbac.PolicyRule{
		// Needed to check API access.  These creates are non-mutating
		rbac.NewRule("create").Groups(authenticationGroup).Resources("tokenreviews").RuleOrDie(),
		rbac.NewRule("create").Groups(authorizationGroup).Resources("subjectaccessreviews", "localsubjectaccessreviews").RuleOrDie(),

		// Needed to build serviceLister, to populate env vars for services
		rbac.NewRule(Read...).Groups(legacyGroup).Resources("services").RuleOrDie(),

		// Nodes can register Node API objects and report status.
		// Use the NodeRestriction admission plugin to limit a node to creating/updating its own API object.
		rbac.NewRule("create", "get", "list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
		rbac.NewRule("update", "patch").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),
		rbac.NewRule("update", "patch", "delete").Groups(legacyGroup).Resources("nodes").RuleOrDie(),

		// TODO: restrict to the bound node as creator in the NodeRestrictions admission plugin
		rbac.NewRule("create", "update", "patch").Groups(legacyGroup).Resources("events").RuleOrDie(),

		// TODO: restrict to pods scheduled on the bound node once field selectors are supported by list/watch authorization
		rbac.NewRule(Read...).Groups(legacyGroup).Resources("pods").RuleOrDie(),

		// Needed for the node to create/delete mirror pods.
		// Use the NodeRestriction admission plugin to limit a node to creating/deleting mirror pods bound to itself.
		rbac.NewRule("create", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
		// Needed for the node to report status of pods it is running.
		// Use the NodeRestriction admission plugin to limit a node to updating status of pods bound to itself.
		rbac.NewRule("update").Groups(legacyGroup).Resources("pods/status").RuleOrDie(),
		// Needed for the node to create pod evictions.
		// Use the NodeRestriction admission plugin to limit a node to creating evictions for pods bound to itself.
		rbac.NewRule("create").Groups(legacyGroup).Resources("pods/eviction").RuleOrDie(),

		// Needed for imagepullsecrets, rbd/ceph and secret volumes, and secrets in envs
		// Needed for configmap volume and envs
		// Use the NodeRestriction admission plugin to limit a node to get secrets/configmaps referenced by pods bound to itself.
		rbac.NewRule("get").Groups(legacyGroup).Resources("secrets", "configmaps").RuleOrDie(),
		// Needed for persistent volumes
		// Use the NodeRestriction admission plugin to limit a node to get pv/pvc objects referenced by pods bound to itself.
		rbac.NewRule("get").Groups(legacyGroup).Resources("persistentvolumeclaims", "persistentvolumes").RuleOrDie(),
		// TODO: add to the Node authorizer and restrict to endpoints referenced by pods or PVs bound to the node
		// Needed for glusterfs volumes
		rbac.NewRule("get").Groups(legacyGroup).Resources("endpoints").RuleOrDie(),
		// Used to create a certificatesigningrequest for a node-specific client certificate, and watch
		// for it to be signed. This allows the kubelet to rotate it's own certificate.
		rbac.NewRule("create", "get", "list", "watch").Groups(certificatesGroup).Resources("certificatesigningrequests").RuleOrDie(),
	}
}

// ClusterRoles returns the cluster roles to bootstrap an API server with
func ClusterRoles() []rbac.ClusterRole {
	roles := []rbac.ClusterRole{
		{
			// a "root" role which can do absolutely anything
			ObjectMeta: metav1.ObjectMeta{Name: "cluster-admin"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("*").Groups("*").Resources("*").RuleOrDie(),
				rbac.NewRule("*").URLs("*").RuleOrDie(),
			},
		},
		{
			// a role which provides just enough power to determine if the server is ready and discover API versions for negotiation
			ObjectMeta: metav1.ObjectMeta{Name: "system:discovery"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("get").URLs("/healthz", "/version", "/swaggerapi", "/swaggerapi/*", "/api", "/api/*", "/apis", "/apis/*").RuleOrDie(),
			},
		},
		{
			// a role which provides minimal resource access to allow a "normal" user to learn information about themselves
			ObjectMeta: metav1.ObjectMeta{Name: "system:basic-user"},
			Rules: []rbac.PolicyRule{
				// TODO add future selfsubjectrulesreview, project request APIs, project listing APIs
				rbac.NewRule("create").Groups(authorizationGroup).Resources("selfsubjectaccessreviews").RuleOrDie(),
			},
		},

		{
			// a role for a namespace level admin.  It is `edit` plus the power to grant permissions to other users.
			ObjectMeta: metav1.ObjectMeta{Name: "admin"},
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

				rbac.NewRule(ReadWrite...).Groups(appsGroup).Resources("statefulsets",
					"deployments", "deployments/scale", "deployments/rollback").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(autoscalingGroup).Resources("horizontalpodautoscalers").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(batchGroup).Resources("jobs", "cronjobs").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(extensionsGroup).Resources("daemonsets",
					"deployments", "deployments/scale", "deployments/rollback", "ingresses",
					"replicasets", "replicasets/scale", "replicationcontrollers/scale").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(policyGroup).Resources("poddisruptionbudgets").RuleOrDie(),

				// additional admin powers
				rbac.NewRule("create").Groups(authorizationGroup).Resources("localsubjectaccessreviews").RuleOrDie(),
				rbac.NewRule(ReadWrite...).Groups(rbacGroup).Resources("roles", "rolebindings").RuleOrDie(),
			},
		},
		{
			// a role for a namespace level editor.  It grants access to all user level actions in a namespace.
			// It does not grant powers for "privileged" resources which are domain of the system: `/status`
			// subresources or `quota`/`limits` which are used to control namespaces
			ObjectMeta: metav1.ObjectMeta{Name: "edit"},
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

				rbac.NewRule(ReadWrite...).Groups(appsGroup).Resources("statefulsets",
					"deployments", "deployments/scale", "deployments/rollback").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(autoscalingGroup).Resources("horizontalpodautoscalers").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(batchGroup).Resources("jobs", "cronjobs").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(extensionsGroup).Resources("daemonsets",
					"deployments", "deployments/scale", "deployments/rollback", "ingresses",
					"replicasets", "replicasets/scale", "replicationcontrollers/scale").RuleOrDie(),

				rbac.NewRule(ReadWrite...).Groups(policyGroup).Resources("poddisruptionbudgets").RuleOrDie(),
			},
		},
		{
			// a role for namespace level viewing.  It grants Read-only access to non-escalating resources in
			// a namespace.
			ObjectMeta: metav1.ObjectMeta{Name: "view"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("pods", "replicationcontrollers", "replicationcontrollers/scale", "serviceaccounts",
					"services", "endpoints", "persistentvolumeclaims", "configmaps").RuleOrDie(),
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("limitranges", "resourcequotas", "bindings", "events",
					"pods/status", "resourcequotas/status", "namespaces/status", "replicationcontrollers/status", "pods/log").RuleOrDie(),
				// read access to namespaces at the namespace scope means you can read *this* namespace.  This can be used as an
				// indicator of which namespaces you have access to.
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("namespaces").RuleOrDie(),

				rbac.NewRule(Read...).Groups(appsGroup).Resources("statefulsets", "deployments", "deployments/scale").RuleOrDie(),

				rbac.NewRule(Read...).Groups(autoscalingGroup).Resources("horizontalpodautoscalers").RuleOrDie(),

				rbac.NewRule(Read...).Groups(batchGroup).Resources("jobs", "cronjobs").RuleOrDie(),

				rbac.NewRule(Read...).Groups(extensionsGroup).Resources("daemonsets", "deployments", "deployments/scale",
					"ingresses", "replicasets", "replicasets/scale", "replicationcontrollers/scale").RuleOrDie(),

				rbac.NewRule(Read...).Groups(policyGroup).Resources("poddisruptionbudgets").RuleOrDie(),
			},
		},
		{
			// a role to use for heapster's connections back to the API server
			ObjectMeta: metav1.ObjectMeta{Name: "system:heapster"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("events", "pods", "nodes", "namespaces").RuleOrDie(),
				rbac.NewRule(Read...).Groups(extensionsGroup).Resources("deployments").RuleOrDie(),
			},
		},
		{
			// a role for nodes to use to have the access they need for running pods
			ObjectMeta: metav1.ObjectMeta{Name: "system:node"},
			Rules:      NodeRules(),
		},
		{
			// a role to use for node-problem-detector access.  It does not get bound to default location since
			// deployment locations can reasonably vary.
			ObjectMeta: metav1.ObjectMeta{Name: "system:node-problem-detector"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("get").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
				rbac.NewRule("patch").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),
				eventsRule(),
			},
		},
		{
			// a role to use for setting up a proxy
			ObjectMeta: metav1.ObjectMeta{Name: "system:node-proxier"},
			Rules: []rbac.PolicyRule{
				// Used to build serviceLister
				rbac.NewRule("list", "watch").Groups(legacyGroup).Resources("services", "endpoints").RuleOrDie(),
				rbac.NewRule("get").Groups(legacyGroup).Resources("nodes").RuleOrDie(),

				eventsRule(),
			},
		},
		{
			// a role to use for bootstrapping a node's client certificates
			ObjectMeta: metav1.ObjectMeta{Name: "system:node-bootstrapper"},
			Rules: []rbac.PolicyRule{
				// used to create a certificatesigningrequest for a node-specific client certificate, and watch for it to be signed
				rbac.NewRule("create", "get", "list", "watch").Groups(certificatesGroup).Resources("certificatesigningrequests").RuleOrDie(),
			},
		},
		{
			// a role to use for allowing authentication and authorization delegation
			ObjectMeta: metav1.ObjectMeta{Name: "system:auth-delegator"},
			Rules: []rbac.PolicyRule{
				// These creates are non-mutating
				rbac.NewRule("create").Groups(authenticationGroup).Resources("tokenreviews").RuleOrDie(),
				rbac.NewRule("create").Groups(authorizationGroup).Resources("subjectaccessreviews").RuleOrDie(),
			},
		},
		{
			// a role to use for the API registry, summarization, and proxy handling
			ObjectMeta: metav1.ObjectMeta{Name: "system:kube-aggregator"},
			Rules: []rbac.PolicyRule{
				// it needs to see all services so that it knows whether the ones it points to exist or not
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("services", "endpoints").RuleOrDie(),
			},
		},
		{
			// a role to use for bootstrapping the kube-controller-manager so it can create the shared informers
			// service accounts, and secrets that we need to create separate identities for other controllers
			ObjectMeta: metav1.ObjectMeta{Name: "system:kube-controller-manager"},
			Rules: []rbac.PolicyRule{
				eventsRule(),
				rbac.NewRule("create").Groups(legacyGroup).Resources("endpoints", "secrets", "serviceaccounts").RuleOrDie(),
				rbac.NewRule("delete").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
				rbac.NewRule("get").Groups(legacyGroup).Resources("endpoints", "namespaces", "secrets", "serviceaccounts").RuleOrDie(),
				rbac.NewRule("update").Groups(legacyGroup).Resources("endpoints", "secrets", "serviceaccounts").RuleOrDie(),
				// Needed to check API access.  These creates are non-mutating
				rbac.NewRule("create").Groups(authenticationGroup).Resources("tokenreviews").RuleOrDie(),
				// Needed for all shared informers
				rbac.NewRule("list", "watch").Groups("*").Resources("*").RuleOrDie(),
			},
		},
		{
			// a role to use for the kube-scheduler
			ObjectMeta: metav1.ObjectMeta{Name: "system:kube-scheduler"},
			Rules: []rbac.PolicyRule{
				eventsRule(),

				// this is for leaderlease access
				// TODO: scope this to the kube-system namespace
				rbac.NewRule("create").Groups(legacyGroup).Resources("endpoints").RuleOrDie(),
				rbac.NewRule("get", "update", "patch", "delete").Groups(legacyGroup).Resources("endpoints").Names("kube-scheduler").RuleOrDie(),

				// fundamental resources
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("nodes").RuleOrDie(),
				rbac.NewRule("get", "list", "watch", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
				rbac.NewRule("create").Groups(legacyGroup).Resources("pods/binding", "bindings").RuleOrDie(),
				rbac.NewRule("update").Groups(legacyGroup).Resources("pods/status").RuleOrDie(),
				// things that select pods
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("services", "replicationcontrollers").RuleOrDie(),
				rbac.NewRule(Read...).Groups(extensionsGroup).Resources("replicasets").RuleOrDie(),
				rbac.NewRule(Read...).Groups(appsGroup).Resources("statefulsets").RuleOrDie(),
				// things that pods use
				rbac.NewRule(Read...).Groups(legacyGroup).Resources("persistentvolumeclaims", "persistentvolumes").RuleOrDie(),
			},
		},
		{
			// a role to use for the kube-dns pod
			ObjectMeta: metav1.ObjectMeta{Name: "system:kube-dns"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("list", "watch").Groups(legacyGroup).Resources("endpoints", "services").RuleOrDie(),
			},
		},
		{
			// a role for an external/out-of-tree persistent volume provisioner
			ObjectMeta: metav1.ObjectMeta{Name: "system:persistent-volume-provisioner"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("get", "list", "watch", "create", "delete").Groups(legacyGroup).Resources("persistentvolumes").RuleOrDie(),
				// update is needed in addition to read access for setting lock annotations on PVCs
				rbac.NewRule("get", "list", "watch", "update").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
				rbac.NewRule(Read...).Groups(storageGroup).Resources("storageclasses").RuleOrDie(),

				// Needed for watching provisioning success and failure events
				rbac.NewRule("watch").Groups(legacyGroup).Resources("events").RuleOrDie(),

				eventsRule(),
			},
		},
		{
			// a role making the csrapprover controller approve a node client CSR
			ObjectMeta: metav1.ObjectMeta{Name: "system:certificates.k8s.io:certificatesigningrequests:nodeclient"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("create").Groups(certificatesGroup).Resources("certificatesigningrequests/nodeclient").RuleOrDie(),
			},
		},
		{
			// a role making the csrapprover controller approve a node client CSR requested by the node itself
			ObjectMeta: metav1.ObjectMeta{Name: "system:certificates.k8s.io:certificatesigningrequests:selfnodeclient"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("create").Groups(certificatesGroup).Resources("certificatesigningrequests/selfnodeclient").RuleOrDie(),
			},
		},
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.RotateKubeletServerCertificate) {
		roles = append(roles, rbac.ClusterRole{
			// a role making the csrapprover controller approve a node server CSR requested by the node itself
			ObjectMeta: metav1.ObjectMeta{Name: "system:certificates.k8s.io:certificatesigningrequests:selfnodeserver"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("create").Groups(certificatesGroup).Resources("certificatesigningrequests/selfnodeserver").RuleOrDie(),
			},
		})
	}

	addClusterRoleLabel(roles)
	return roles
}

const systemNodeRoleName = "system:node"

// ClusterRoleBindings return default rolebindings to the default roles
func ClusterRoleBindings() []rbac.ClusterRoleBinding {
	rolebindings := []rbac.ClusterRoleBinding{
		rbac.NewClusterBinding("cluster-admin").Groups(user.SystemPrivilegedGroup).BindingOrDie(),
		rbac.NewClusterBinding("system:discovery").Groups(user.AllAuthenticated, user.AllUnauthenticated).BindingOrDie(),
		rbac.NewClusterBinding("system:basic-user").Groups(user.AllAuthenticated, user.AllUnauthenticated).BindingOrDie(),
		rbac.NewClusterBinding("system:node-proxier").Users(user.KubeProxy).BindingOrDie(),
		rbac.NewClusterBinding("system:kube-controller-manager").Users(user.KubeControllerManager).BindingOrDie(),
		rbac.NewClusterBinding("system:kube-dns").SAs("kube-system", "kube-dns").BindingOrDie(),
		rbac.NewClusterBinding("system:kube-scheduler").Users(user.KubeScheduler).BindingOrDie(),

		// This default binding of the system:node role to the system:nodes group is deprecated in 1.7 with the availability of the Node authorizer.
		// This leaves the binding, but with an empty set of subjects, so that tightening reconciliation can remove the subject.
		{
			ObjectMeta: metav1.ObjectMeta{Name: systemNodeRoleName},
			RoleRef:    rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: systemNodeRoleName},
		},
	}

	addClusterRoleBindingLabel(rolebindings)

	return rolebindings
}
