package customresourcevalidationregistration

import (
	"k8s.io/apiserver/pkg/admission"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/apiserver"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/authentication"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/clusterresourcequota"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/config"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/console"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/features"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/image"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/oauth"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/project"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/rolebindingrestriction"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/scheduler"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/securitycontextconstraints"
)

// AllCustomResourceValidators are the names of all custom resource validators that should be registered
var AllCustomResourceValidators = []string{
	apiserver.PluginName,
	authentication.PluginName,
	features.PluginName,
	console.PluginName,
	image.PluginName,
	oauth.PluginName,
	project.PluginName,
	config.PluginName,
	scheduler.PluginName,
	clusterresourcequota.PluginName,
	securitycontextconstraints.PluginName,
	rolebindingrestriction.PluginName,

	// this one is special because we don't work without it.
	securitycontextconstraints.DefaultingPluginName,
}

func RegisterCustomResourceValidation(plugins *admission.Plugins) {
	apiserver.Register(plugins)
	authentication.Register(plugins)
	features.Register(plugins)
	console.Register(plugins)
	image.Register(plugins)
	oauth.Register(plugins)
	project.Register(plugins)
	config.Register(plugins)
	scheduler.Register(plugins)

	// This plugin validates the quota.openshift.io/v1 ClusterResourceQuota resources.
	// NOTE: This is only allowed because it is required to get a running control plane operator.
	clusterresourcequota.Register(plugins)
	// This plugin validates the security.openshift.io/v1 SecurityContextConstraints resources.
	securitycontextconstraints.Register(plugins)
	// This plugin validates the authorization.openshift.io/v1 RoleBindingRestriction resources.
	rolebindingrestriction.Register(plugins)

	// this one is special because we don't work without it.
	securitycontextconstraints.RegisterDefaulting(plugins)
}
