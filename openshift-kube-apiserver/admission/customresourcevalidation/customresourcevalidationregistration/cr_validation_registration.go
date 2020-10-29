package customresourcevalidationregistration

import (
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/apirequestcount"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/apiserver"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/authentication"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/clusterresourcequota"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/config"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/console"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/dns"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/features"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/image"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/kubecontrollermanager"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/network"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/node"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/oauth"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/operator"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/project"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/rolebindingrestriction"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/route"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/scheduler"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/securitycontextconstraints"
)

// AllCustomResourceValidators are the names of all custom resource validators that should be registered
var AllCustomResourceValidators = []string{
	apiserver.PluginName,
	authentication.PluginName,
	features.PluginName,
	console.PluginName,
	dns.PluginName,
	image.PluginName,
	oauth.PluginName,
	project.PluginName,
	config.PluginName,
	operator.PluginName,
	scheduler.PluginName,
	clusterresourcequota.PluginName,
	securitycontextconstraints.PluginName,
	rolebindingrestriction.PluginName,
	network.PluginName,
	apirequestcount.PluginName,
	node.PluginName,
	route.DefaultingPluginName,
	route.PluginName,

	// the kubecontrollermanager operator resource has to exist in order to run deployments to deploy admission webhooks.
	kubecontrollermanager.PluginName,

	// this one is special because we don't work without it.
	securitycontextconstraints.DefaultingPluginName,
}

func RegisterCustomResourceValidation(plugins *admission.Plugins) {
	apiserver.Register(plugins)
	authentication.Register(plugins)
	features.Register(plugins)
	console.Register(plugins)
	dns.Register(plugins)
	image.Register(plugins)
	oauth.Register(plugins)
	project.Register(plugins)
	config.Register(plugins)
	operator.Register(plugins)
	scheduler.Register(plugins)
	kubecontrollermanager.Register(plugins)

	// This plugin validates the quota.openshift.io/v1 ClusterResourceQuota resources.
	// NOTE: This is only allowed because it is required to get a running control plane operator.
	clusterresourcequota.Register(plugins)
	// This plugin validates the security.openshift.io/v1 SecurityContextConstraints resources.
	securitycontextconstraints.Register(plugins)
	// This plugin validates the authorization.openshift.io/v1 RoleBindingRestriction resources.
	rolebindingrestriction.Register(plugins)
	// This plugin validates the network.config.openshift.io object for service node port range changes
	network.Register(plugins)
	// This plugin validates the apiserver.openshift.io/v1 APIRequestCount resources.
	apirequestcount.Register(plugins)
	// This plugin validates config.openshift.io/v1/node objects
	node.Register(plugins)

	// this one is special because we don't work without it.
	securitycontextconstraints.RegisterDefaulting(plugins)

	// Requests to route.openshift.io/v1 should only go through kube-apiserver admission if
	// served via CRD. Most OpenShift flavors (including vanilla) will continue to do validation
	// and defaulting inside openshift-apiserver.
	route.Register(plugins)
	route.RegisterDefaulting(plugins)
}
