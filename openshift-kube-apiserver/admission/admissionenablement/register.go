package admissionenablement

import (
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/resourcequota"
	mutatingwebhook "k8s.io/apiserver/pkg/admission/plugin/webhook/mutating"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/mixedcpus"

	"github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy"
	imagepolicyapiv1 "github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/v1"
	quotaclusterresourcequota "github.com/openshift/apiserver-library-go/pkg/admission/quota/clusterresourcequota"
	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccadmission"
	authorizationrestrictusers "k8s.io/kubernetes/openshift-kube-apiserver/admission/authorization/restrictusers"
	quotaclusterresourceoverride "k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/clusterresourceoverride"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/managednode"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/managementcpusoverride"
	quotarunonceduration "k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/runonceduration"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/customresourcevalidationregistration"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/network/externalipranger"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/network/restrictedendpoints"
	ingressadmission "k8s.io/kubernetes/openshift-kube-apiserver/admission/route"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/route/hostassignment"
	projectnodeenv "k8s.io/kubernetes/openshift-kube-apiserver/admission/scheduler/nodeenv"
	schedulerpodnodeconstraints "k8s.io/kubernetes/openshift-kube-apiserver/admission/scheduler/podnodeconstraints"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/storage/csiinlinevolumesecurity"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/storage/performantsecuritypolicy"
)

func RegisterOpenshiftKubeAdmissionPlugins(plugins *admission.Plugins) {
	authorizationrestrictusers.Register(plugins)
	hostassignment.Register(plugins)
	imagepolicy.Register(plugins)
	ingressadmission.Register(plugins)
	managementcpusoverride.Register(plugins)
	managednode.Register(plugins)
	mixedcpus.Register(plugins)
	projectnodeenv.Register(plugins)
	quotaclusterresourceoverride.Register(plugins)
	quotaclusterresourcequota.Register(plugins)
	quotarunonceduration.Register(plugins)
	schedulerpodnodeconstraints.Register(plugins)
	sccadmission.Register(plugins)
	sccadmission.RegisterSCCExecRestrictions(plugins)
	externalipranger.RegisterExternalIP(plugins)
	restrictedendpoints.RegisterRestrictedEndpoints(plugins)
	csiinlinevolumesecurity.Register(plugins)
	performantsecuritypolicy.Register(plugins)
}

var (

	// these are admission plugins that cannot be applied until after the kubeapiserver starts.
	// TODO if nothing comes to mind in 3.10, kill this
	SkipRunLevelZeroPlugins = sets.NewString()
	// these are admission plugins that cannot be applied until after the openshiftapiserver apiserver starts.
	SkipRunLevelOnePlugins = sets.NewString(
		imagepolicyapiv1.PluginName, // "image.openshift.io/ImagePolicy"
		"quota.openshift.io/ClusterResourceQuota",
		"security.openshift.io/SecurityContextConstraint",
		"security.openshift.io/SCCExecRestrictions",
	)

	// openshiftAdmissionPluginsForKubeBeforeMutating are the admission plugins to add after kube admission, before mutating webhooks
	openshiftAdmissionPluginsForKubeBeforeMutating = []string{
		"autoscaling.openshift.io/ClusterResourceOverride",
		managementcpusoverride.PluginName, // "autoscaling.openshift.io/ManagementCPUsOverride"
		"authorization.openshift.io/RestrictSubjectBindings",
		"autoscaling.openshift.io/RunOnceDuration",
		"scheduling.openshift.io/PodNodeConstraints",
		"scheduling.openshift.io/OriginPodNodeEnvironment",
		"network.openshift.io/ExternalIPRanger",
		"network.openshift.io/RestrictedEndpointsAdmission",
		imagepolicyapiv1.PluginName, // "image.openshift.io/ImagePolicy"
		"security.openshift.io/SecurityContextConstraint",
		"security.openshift.io/SCCExecRestrictions",
		"route.openshift.io/IngressAdmission",
		hostassignment.PluginName,           // "route.openshift.io/RouteHostAssignment"
		csiinlinevolumesecurity.PluginName,  // "storage.openshift.io/CSIInlineVolumeSecurity"
		managednode.PluginName,              // "autoscaling.openshift.io/ManagedNode"
		mixedcpus.PluginName,                // "autoscaling.openshift.io/MixedCPUs"
		performantsecuritypolicy.PluginName, // "storage.openshift.io/PerformantSecurityPolicy"
	}

	// openshiftAdmissionPluginsForKubeAfterResourceQuota are the plugins to add after ResourceQuota plugin
	openshiftAdmissionPluginsForKubeAfterResourceQuota = []string{
		"quota.openshift.io/ClusterResourceQuota",
	}

	// additionalDefaultOnPlugins is a list of plugins we turn on by default that core kube does not.
	additionalDefaultOnPlugins = sets.NewString(
		"NodeRestriction",
		"OwnerReferencesPermissionEnforcement",
		"PodNodeSelector",
		"PodTolerationRestriction",
		"Priority",
		imagepolicyapiv1.PluginName, // "image.openshift.io/ImagePolicy"
		"StorageObjectInUseProtection",
	)
)

func NewOrderedKubeAdmissionPlugins(kubeAdmissionOrder []string) []string {
	ret := []string{}
	for _, curr := range kubeAdmissionOrder {
		if curr == mutatingwebhook.PluginName {
			ret = append(ret, openshiftAdmissionPluginsForKubeBeforeMutating...)
			ret = append(ret, customresourcevalidationregistration.AllCustomResourceValidators...)
		}

		ret = append(ret, curr)

		if curr == resourcequota.PluginName {
			ret = append(ret, openshiftAdmissionPluginsForKubeAfterResourceQuota...)
		}
	}
	return ret
}

func NewDefaultOffPluginsFunc(kubeDefaultOffAdmission sets.Set[string]) func() sets.Set[string] {
	return func() sets.Set[string] {
		kubeOff := sets.New[string](kubeDefaultOffAdmission.UnsortedList()...)
		kubeOff.Delete(additionalDefaultOnPlugins.List()...)
		kubeOff.Delete(openshiftAdmissionPluginsForKubeBeforeMutating...)
		kubeOff.Delete(openshiftAdmissionPluginsForKubeAfterResourceQuota...)
		kubeOff.Delete(customresourcevalidationregistration.AllCustomResourceValidators...)
		return kubeOff
	}
}
