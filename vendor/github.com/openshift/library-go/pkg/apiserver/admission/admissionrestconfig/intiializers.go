package admissionrestconfig

import (
	"k8s.io/apiserver/pkg/admission"
	restclient "k8s.io/client-go/rest"
)

func NewInitializer(restClientConfig restclient.Config) admission.PluginInitializer {
	return &localInitializer{
		restClientConfig: restClientConfig,
	}
}

// WantsRESTClientConfig gives access to a RESTClientConfig.  It's useful for doing unusual things with transports.
type WantsRESTClientConfig interface {
	SetRESTClientConfig(restclient.Config)
	admission.InitializationValidator
}

type localInitializer struct {
	restClientConfig restclient.Config
}

// Initialize will check the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *localInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsRESTClientConfig); ok {
		wants.SetRESTClientConfig(i.restClientConfig)
	}
}
