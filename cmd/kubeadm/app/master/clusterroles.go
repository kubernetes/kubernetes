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
		ObjectMeta: api.ObjectMeta{Name: "kubeadm:kubelet-bootstrap"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get").Groups("").Resources("nodes").RuleOrDie(),
			rbac.NewRule("create", "watch").Groups("certificates.k8s.io").Resources("certificatesigningrequests").RuleOrDie(),
		},
	}
	if _, err := clientset.ClusterRoles().Create(&clusterRole); err != nil {
		return err
	}

	subject := rbac.Subject{
		Kind: "Group",
		Name: "kubeadm:kubelet-bootstrap",
	}

	clusterRoleBinding := rbac.ClusterRoleBinding{
		ObjectMeta: api.ObjectMeta{
			Name: "kubeadm:kubelet-bootstrap",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "kubeadm:kubelet-bootstrap",
		},
		Subjects: []rbac.Subject{subject},
	}
	if _, err := clientset.ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
		return err
	}
	fmt.Println("[apiconfig] Created kubelet-bootstrap RBAC rules")

	return nil
}

// CreateKubeDNSRBACClusterRole creates the necessary ClusterRole for kube-dns
func CreateKubeDNSRBACClusterRole(clientset *clientset.Clientset) error {
	clusterRole := rbac.ClusterRole{
		ObjectMeta: api.ObjectMeta{Name: "kubeadm:kube-dns"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("list", "watch").Groups("").Resources("endpoints", "services", "configmaps").RuleOrDie(),
			// TODO: remove this once https://github.com/kubernetes/kubernetes/pull/38816 is merged
			rbac.NewRule("list", "watch").Groups("").Resources("configmaps").RuleOrDie(),
			rbac.NewRule("get").Groups("").Resources("configmaps").RuleOrDie(),
		},
	}
	if _, err := clientset.ClusterRoles().Create(&clusterRole); err != nil {
		return err
	}

	subject := rbac.Subject{
		Kind:      "ServiceAccount",
		Name:      "kube-dns",
		Namespace: "kube-system",
	}

	clusterRoleBinding := rbac.ClusterRoleBinding{
		ObjectMeta: api.ObjectMeta{
			Name: "kubeadm:kube-dns",
		},
		RoleRef: rbac.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "kubeadm:kube-dns",
		},
		Subjects: []rbac.Subject{subject},
	}
	if _, err := clientset.ClusterRoleBindings().Create(&clusterRoleBinding); err != nil {
		return err
	}
	fmt.Println("[apiconfig] Created kube-dns RBAC rules")

	return nil
}
