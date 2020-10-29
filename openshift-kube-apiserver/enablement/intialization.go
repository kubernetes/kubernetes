package enablement

import (
	"io/ioutil"
	"path"

	configv1 "github.com/openshift/api/config/v1"
	kubecontrolplanev1 "github.com/openshift/api/kubecontrolplane/v1"
	osinv1 "github.com/openshift/api/osin/v1"
	"github.com/openshift/library-go/pkg/config/helpers"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/clientcmd/api"
	aggregatorapiserver "k8s.io/kube-aggregator/pkg/apiserver"
	"k8s.io/kubernetes/openshift-kube-apiserver/configdefault"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

func GetOpenshiftConfig(openshiftConfigFile string) (*kubecontrolplanev1.KubeAPIServerConfig, error) {
	// try to decode into our new types first.  right now there is no validation, no file path resolution.  this unsticks the operator to start.
	// TODO add those things
	configContent, err := ioutil.ReadFile(openshiftConfigFile)
	if err != nil {
		return nil, err
	}
	scheme := runtime.NewScheme()
	utilruntime.Must(kubecontrolplanev1.Install(scheme))
	codecs := serializer.NewCodecFactory(scheme)
	obj, err := runtime.Decode(codecs.UniversalDecoder(kubecontrolplanev1.GroupVersion, configv1.GroupVersion, osinv1.GroupVersion), configContent)
	if err != nil {

		return nil, err
	}

	// Resolve relative to CWD
	absoluteConfigFile, err := api.MakeAbs(openshiftConfigFile, "")
	if err != nil {
		return nil, err
	}
	configFileLocation := path.Dir(absoluteConfigFile)

	config := obj.(*kubecontrolplanev1.KubeAPIServerConfig)
	if err := helpers.ResolvePaths(configdefault.GetKubeAPIServerConfigFileReferences(config), configFileLocation); err != nil {
		return nil, err
	}
	configdefault.SetRecommendedKubeAPIServerConfigDefaults(config)
	configdefault.ResolveDirectoriesForSATokenVerification(config)

	return config, nil
}

func ForceGlobalInitializationForOpenShift() {
	// This allows to move crqs, sccs, and rbrs to CRD
	aggregatorapiserver.AddAlwaysLocalDelegateForPrefix("/apis/quota.openshift.io/v1/clusterresourcequotas")
	aggregatorapiserver.AddAlwaysLocalDelegateForPrefix("/apis/security.openshift.io/v1/securitycontextconstraints")
	aggregatorapiserver.AddAlwaysLocalDelegateForPrefix("/apis/authorization.openshift.io/v1/rolebindingrestrictions")
	aggregatorapiserver.AddAlwaysLocalDelegateGroupResource(schema.GroupResource{Group: "authorization.openshift.io", Resource: "rolebindingrestrictions"})

	// This allows the CRD registration to avoid fighting with the APIService from the operator
	aggregatorapiserver.AddOverlappingGroupVersion(schema.GroupVersion{Group: "authorization.openshift.io", Version: "v1"})

	// Allow privileged containers
	capabilities.Initialize(capabilities.Capabilities{
		AllowPrivileged: true,
		PrivilegedSources: capabilities.PrivilegedSources{
			HostNetworkSources: []string{kubelettypes.ApiserverSource, kubelettypes.FileSource},
			HostPIDSources:     []string{kubelettypes.ApiserverSource, kubelettypes.FileSource},
			HostIPCSources:     []string{kubelettypes.ApiserverSource, kubelettypes.FileSource},
		},
	})

	// add permissions we require on our kube-apiserver
	// TODO, we should scrub these out
	bootstrappolicy.ClusterRoles = bootstrappolicy.OpenshiftClusterRoles
	bootstrappolicy.ClusterRoleBindings = bootstrappolicy.OpenshiftClusterRoleBindings

	// we need to have the authorization chain place something before system:masters
	// SkipSystemMastersAuthorizer disable implicitly added system/master authz, and turn it into another authz mode "SystemMasters", to be added via authorization-mode
	authorizer.SkipSystemMastersAuthorizer()
}
