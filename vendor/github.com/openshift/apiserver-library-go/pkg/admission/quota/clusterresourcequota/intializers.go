package clusterresourcequota

import (
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"

	quotainformer "github.com/openshift/client-go/quota/informers/externalversions/quota/v1"
	"github.com/openshift/library-go/pkg/quota/clusterquotamapping"
)

func NewInitializer(
	clusterResourceQuotaInformer quotainformer.ClusterResourceQuotaInformer,
	clusterQuotaMapper clusterquotamapping.ClusterQuotaMapper,
	quotaRegistry quota.Registry,
) admission.PluginInitializer {
	return &localInitializer{
		clusterResourceQuotaInformer: clusterResourceQuotaInformer,
		clusterQuotaMapper:           clusterQuotaMapper,
		quotaRegistry:                quotaRegistry,
	}
}

// WantsClusterQuota should be implemented by admission plugins that need to know how to map between
// cluster quota and namespaces and get access to the informer.
type WantsClusterQuota interface {
	SetClusterQuota(clusterquotamapping.ClusterQuotaMapper, quotainformer.ClusterResourceQuotaInformer)
	admission.InitializationValidator
}

// WantsQuotaRegistry should be implemented by admission plugins that need a quota registry
type WantsOriginQuotaRegistry interface {
	SetOriginQuotaRegistry(quota.Registry)
	admission.InitializationValidator
}

type localInitializer struct {
	clusterResourceQuotaInformer quotainformer.ClusterResourceQuotaInformer
	clusterQuotaMapper           clusterquotamapping.ClusterQuotaMapper
	quotaRegistry                quota.Registry
}

// Initialize will check the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *localInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsClusterQuota); ok {
		wants.SetClusterQuota(i.clusterQuotaMapper, i.clusterResourceQuotaInformer)
	}
	if wants, ok := plugin.(WantsOriginQuotaRegistry); ok {
		wants.SetOriginQuotaRegistry(i.quotaRegistry)
	}
}
