package openshiftkubeapiserver

import (
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	authorizerunion "k8s.io/apiserver/pkg/authorization/union"
	"k8s.io/client-go/informers"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/node"
	rbacauthorizer "k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac"
	kbootstrappolicy "k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"

	"k8s.io/kubernetes/openshift-kube-apiserver/authorization/browsersafe"
	"k8s.io/kubernetes/openshift-kube-apiserver/authorization/scopeauthorizer"
)

func NewAuthorizer(versionedInformers informers.SharedInformerFactory) authorizer.Authorizer {
	rbacInformers := versionedInformers.Rbac().V1()

	scopeLimitedAuthorizer := scopeauthorizer.NewAuthorizer(rbacInformers.ClusterRoles().Lister())

	kubeAuthorizer := rbacauthorizer.New(
		&rbacauthorizer.RoleGetter{Lister: rbacInformers.Roles().Lister()},
		&rbacauthorizer.RoleBindingLister{Lister: rbacInformers.RoleBindings().Lister()},
		&rbacauthorizer.ClusterRoleGetter{Lister: rbacInformers.ClusterRoles().Lister()},
		&rbacauthorizer.ClusterRoleBindingLister{Lister: rbacInformers.ClusterRoleBindings().Lister()},
	)

	graph := node.NewGraph()
	node.AddGraphEventHandlers(
		graph,
		versionedInformers.Core().V1().Nodes(),
		versionedInformers.Core().V1().Pods(),
		versionedInformers.Core().V1().PersistentVolumes(),
		versionedInformers.Storage().V1().VolumeAttachments(),
	)
	nodeAuthorizer := node.NewAuthorizer(graph, nodeidentifier.NewDefaultNodeIdentifier(), kbootstrappolicy.NodeRules())

	openshiftAuthorizer := authorizerunion.New(
		// Wrap with an authorizer that detects unsafe requests and modifies verbs/resources appropriately so policy can address them separately.
		// Scopes are first because they will authoritatively deny and can logically be attached to anyone.
		browsersafe.NewBrowserSafeAuthorizer(scopeLimitedAuthorizer, user.AllAuthenticated),
		// authorizes system:masters to do anything, just like upstream
		authorizerfactory.NewPrivilegedGroups(user.SystemPrivilegedGroup),
		nodeAuthorizer,
		// Wrap with an authorizer that detects unsafe requests and modifies verbs/resources appropriately so policy can address them separately
		browsersafe.NewBrowserSafeAuthorizer(kubeAuthorizer, user.AllAuthenticated),
	)

	return openshiftAuthorizer
}
