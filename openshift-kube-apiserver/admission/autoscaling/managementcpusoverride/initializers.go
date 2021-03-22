package managementcpusoverride

import (
	"k8s.io/apiserver/pkg/admission"

	configv1informer "github.com/openshift/client-go/config/informers/externalversions/config/v1"
)

func NewInitializer(infraInformer configv1informer.InfrastructureInformer) admission.PluginInitializer {
	return &localInitializer{infraInformer: infraInformer}
}

type WantsInfraInformer interface {
	SetInfraInformer(informer configv1informer.InfrastructureInformer)
	admission.InitializationValidator
}

type localInitializer struct {
	infraInformer configv1informer.InfrastructureInformer
}

// Initialize will check the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *localInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsInfraInformer); ok {
		wants.SetInfraInformer(i.infraInformer)
	}
}
