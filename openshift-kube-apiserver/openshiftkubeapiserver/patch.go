package openshiftkubeapiserver

import (
	"time"

	"k8s.io/kubernetes/openshift-kube-apiserver/enablement"

	"k8s.io/apiserver/pkg/admission"
	genericapiserver "k8s.io/apiserver/pkg/server"
	clientgoinformers "k8s.io/client-go/informers"
	corev1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/quota/v1/generic"
	"k8s.io/kubernetes/pkg/quota/v1/install"

	kubecontrolplanev1 "github.com/openshift/api/kubecontrolplane/v1"
	"github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy"
	"github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/imagereferencemutators"
	"github.com/openshift/apiserver-library-go/pkg/admission/quota/clusterresourcequota"
	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccadmission"
	quotaclient "github.com/openshift/client-go/quota/clientset/versioned"
	quotainformer "github.com/openshift/client-go/quota/informers/externalversions"
	quotav1informer "github.com/openshift/client-go/quota/informers/externalversions/quota/v1"
	securityv1client "github.com/openshift/client-go/security/clientset/versioned"
	securityv1informer "github.com/openshift/client-go/security/informers/externalversions"
	userclient "github.com/openshift/client-go/user/clientset/versioned"
	userinformer "github.com/openshift/client-go/user/informers/externalversions"
	"github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig"
	"github.com/openshift/library-go/pkg/apiserver/apiserverconfig"
	"github.com/openshift/library-go/pkg/quota/clusterquotamapping"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/authorization/restrictusers"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/authorization/restrictusers/usercache"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/scheduler/nodeenv"
)

type KubeAPIServerConfigFunc func(config *genericapiserver.Config, versionedInformers clientgoinformers.SharedInformerFactory, pluginInitializers *[]admission.PluginInitializer) error

func NewOpenShiftKubeAPIServerConfigPatch(kubeAPIServerConfig *kubecontrolplanev1.KubeAPIServerConfig) KubeAPIServerConfigFunc {
	return func(genericConfig *genericapiserver.Config, kubeInformers clientgoinformers.SharedInformerFactory, pluginInitializers *[]admission.PluginInitializer) error {
		openshiftInformers, err := newInformers(genericConfig.LoopbackClientConfig)
		if err != nil {
			return err
		}

		// AUTHORIZER
		genericConfig.RequestInfoResolver = apiserverconfig.OpenshiftRequestInfoResolver()
		// END AUTHORIZER

		// Inject OpenShift API long running endpoints (like for binary builds).
		// TODO: We should disable the timeout code for aggregated endpoints as this can cause problems when upstream add additional endpoints.
		genericConfig.LongRunningFunc = apiserverconfig.IsLongRunningRequest

		// ADMISSION
		clusterQuotaMappingController := newClusterQuotaMappingController(kubeInformers.Core().V1().Namespaces(), openshiftInformers.OpenshiftQuotaInformers.Quota().V1().ClusterResourceQuotas())
		genericConfig.AddPostStartHookOrDie("quota.openshift.io-clusterquotamapping", func(context genericapiserver.PostStartHookContext) error {
			go clusterQuotaMappingController.Run(5, context.StopCh)
			return nil
		})

		*pluginInitializers = append(*pluginInitializers,
			imagepolicy.NewInitializer(imagereferencemutators.KubeImageMutators{}, kubeAPIServerConfig.ImagePolicyConfig.InternalRegistryHostname),
			restrictusers.NewInitializer(openshiftInformers.getOpenshiftUserInformers()),
			sccadmission.NewInitializer(openshiftInformers.getOpenshiftSecurityInformers().Security().V1().SecurityContextConstraints()),
			clusterresourcequota.NewInitializer(
				openshiftInformers.getOpenshiftQuotaInformers().Quota().V1().ClusterResourceQuotas(),
				clusterQuotaMappingController.GetClusterQuotaMapper(),
				generic.NewRegistry(install.NewQuotaConfigurationForAdmission().Evaluators()),
			),
			nodeenv.NewInitializer(kubeAPIServerConfig.ProjectConfig.DefaultNodeSelector),
			admissionrestconfig.NewInitializer(*rest.CopyConfig(genericConfig.LoopbackClientConfig)),
		)
		// END ADMISSION

		// HANDLER CHAIN (with oauth server and web console)
		genericConfig.BuildHandlerChainFunc, err = BuildHandlerChain(kubeAPIServerConfig.ConsolePublicURL, kubeAPIServerConfig.AuthConfig.OAuthMetadataFile)
		if err != nil {
			return err
		}
		// END HANDLER CHAIN

		genericConfig.AddPostStartHookOrDie("openshift.io-startkubeinformers", func(context genericapiserver.PostStartHookContext) error {
			go kubeInformers.Start(context.StopCh)
			go openshiftInformers.Start(context.StopCh)
			return nil
		})
		enablement.AppendPostStartHooksOrDie(genericConfig)

		return nil
	}
}

// newInformers is only exposed for the build's integration testing until it can be fixed more appropriately.
func newInformers(loopbackClientConfig *rest.Config) (*kubeAPIServerInformers, error) {
	// ClusterResourceQuota is served using CRD resource any status update must use JSON
	jsonLoopbackClientConfig := rest.CopyConfig(loopbackClientConfig)
	jsonLoopbackClientConfig.ContentConfig.AcceptContentTypes = "application/json"
	jsonLoopbackClientConfig.ContentConfig.ContentType = "application/json"

	quotaClient, err := quotaclient.NewForConfig(jsonLoopbackClientConfig)
	if err != nil {
		return nil, err
	}
	securityClient, err := securityv1client.NewForConfig(jsonLoopbackClientConfig)
	if err != nil {
		return nil, err
	}
	userClient, err := userclient.NewForConfig(loopbackClientConfig)
	if err != nil {
		return nil, err
	}

	// TODO find a single place to create and start informers.  During the 1.7 rebase this will come more naturally in a config object,
	// before then we should try to eliminate our direct to storage access.  It's making us do weird things.
	const defaultInformerResyncPeriod = 10 * time.Minute

	ret := &kubeAPIServerInformers{
		OpenshiftQuotaInformers:    quotainformer.NewSharedInformerFactory(quotaClient, defaultInformerResyncPeriod),
		OpenshiftSecurityInformers: securityv1informer.NewSharedInformerFactory(securityClient, defaultInformerResyncPeriod),
		OpenshiftUserInformers:     userinformer.NewSharedInformerFactory(userClient, defaultInformerResyncPeriod),
	}
	if err := ret.OpenshiftUserInformers.User().V1().Groups().Informer().AddIndexers(cache.Indexers{
		usercache.ByUserIndexName: usercache.ByUserIndexKeys,
	}); err != nil {
		return nil, err
	}

	return ret, nil
}

type kubeAPIServerInformers struct {
	OpenshiftQuotaInformers    quotainformer.SharedInformerFactory
	OpenshiftSecurityInformers securityv1informer.SharedInformerFactory
	OpenshiftUserInformers     userinformer.SharedInformerFactory
}

func (i *kubeAPIServerInformers) getOpenshiftQuotaInformers() quotainformer.SharedInformerFactory {
	return i.OpenshiftQuotaInformers
}
func (i *kubeAPIServerInformers) getOpenshiftSecurityInformers() securityv1informer.SharedInformerFactory {
	return i.OpenshiftSecurityInformers
}
func (i *kubeAPIServerInformers) getOpenshiftUserInformers() userinformer.SharedInformerFactory {
	return i.OpenshiftUserInformers
}

func (i *kubeAPIServerInformers) Start(stopCh <-chan struct{}) {
	i.OpenshiftQuotaInformers.Start(stopCh)
	i.OpenshiftSecurityInformers.Start(stopCh)
	i.OpenshiftUserInformers.Start(stopCh)
}

func newClusterQuotaMappingController(nsInternalInformer corev1informers.NamespaceInformer, clusterQuotaInformer quotav1informer.ClusterResourceQuotaInformer) *clusterquotamapping.ClusterQuotaMappingController {
	return clusterquotamapping.NewClusterQuotaMappingController(nsInternalInformer, clusterQuotaInformer)
}
