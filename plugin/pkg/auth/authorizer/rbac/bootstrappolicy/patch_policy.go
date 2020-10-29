package bootstrappolicy

import (
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

var ClusterRoles = clusterRoles

func OpenshiftClusterRoles() []rbacv1.ClusterRole {
	const (
		// These are valid under the "nodes" resource
		NodeMetricsSubresource = "metrics"
		NodeStatsSubresource   = "stats"
		NodeSpecSubresource    = "spec"
		NodeLogSubresource     = "log"
	)

	roles := clusterRoles()
	roles = append(roles, []rbacv1.ClusterRole{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "system:node-admin",
			},
			Rules: []rbacv1.PolicyRule{
				// Allow read-only access to the API objects
				rbacv1helpers.NewRule(Read...).Groups(legacyGroup).Resources("nodes").RuleOrDie(),
				// Allow all API calls to the nodes
				rbacv1helpers.NewRule("proxy").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
				rbacv1helpers.NewRule("*").Groups(legacyGroup).Resources("nodes/proxy", "nodes/"+NodeMetricsSubresource, "nodes/"+NodeSpecSubresource, "nodes/"+NodeStatsSubresource, "nodes/"+NodeLogSubresource).RuleOrDie(),
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "system:node-reader",
			},
			Rules: []rbacv1.PolicyRule{
				// Allow read-only access to the API objects
				rbacv1helpers.NewRule(Read...).Groups(legacyGroup).Resources("nodes").RuleOrDie(),
				// Allow read access to node metrics
				rbacv1helpers.NewRule("get").Groups(legacyGroup).Resources("nodes/"+NodeMetricsSubresource, "nodes/"+NodeSpecSubresource).RuleOrDie(),
				// Allow read access to stats
				// Node stats requests are submitted as POSTs.  These creates are non-mutating
				rbacv1helpers.NewRule("get", "create").Groups(legacyGroup).Resources("nodes/" + NodeStatsSubresource).RuleOrDie(),
				// TODO: expose other things like /healthz on the node once we figure out non-resource URL policy across systems
			},
		},
	}...)

	addClusterRoleLabel(roles)
	return roles
}

var ClusterRoleBindings = clusterRoleBindings

func OpenshiftClusterRoleBindings() []rbacv1.ClusterRoleBinding {
	bindings := clusterRoleBindings()
	bindings = append(bindings, []rbacv1.ClusterRoleBinding{
		rbacv1helpers.NewClusterBinding("system:node-admin").Users("system:master", "system:kube-apiserver").Groups("system:node-admins").BindingOrDie(),
	}...)

	addClusterRoleBindingLabel(bindings)
	return bindings
}
