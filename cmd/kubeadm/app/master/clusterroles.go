package master

import (
	"fmt"

	api "k8s.io/kubernetes/pkg/api/v1"
	rbac "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// CreateBootstrapRBACClusterRole creates the necessary ClusterRole for bootstrapping
func CreateBootstrapRBACClusterRole(clientset *clientset.Clientset) error {
	clusterRole := rbac.ClusterRole{
		// a role to use for setting up a proxy
		ObjectMeta: api.ObjectMeta{Name: "system:kubelet-bootstrap"},
		Rules: []rbac.PolicyRule{
			// Used to build serviceLister
			rbac.NewRule("get").Groups("").Resources("nodes").RuleOrDie(),
			rbac.NewRule("create", "watch").Groups("certificates.k8s.io").Resources("certificatesigningrequests").RuleOrDie(),
		},
	}
	if _, err := clientset.ClusterRoles().Create(&clusterRole); err != nil {
		return err
	}
	fmt.Println("[clusterroles] Created RBAC ClusterRole")

	subject := rbac.Subject{
		Kind: "Group",
		Name: "system:kubelet-bootstrap",
	}

	clusterRoleBinding := rbac.ClusterRoleBinding{
		ObjectMeta: api.ObjectMeta{
			Name: "system:kubelet-bootstrap",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "system:kubelet-bootstrap",
		},
		Subjects: []rbac.Subject{subject},
	}
	if _, err := clientset.ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
		return err
	}
	fmt.Println("[clusterrolebindings] Created RBAC ClusterRoleBinding")

	return nil
}
