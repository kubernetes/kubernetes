package nodeenv

import (
	"k8s.io/apiserver/pkg/admission"
)

func NewInitializer(defaultNodeSelector string) admission.PluginInitializer {
	return &localInitializer{
		defaultNodeSelector: defaultNodeSelector,
	}
}

type WantsDefaultNodeSelector interface {
	SetDefaultNodeSelector(string)
	admission.InitializationValidator
}

type localInitializer struct {
	defaultNodeSelector string
}

// Initialize will check the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *localInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsDefaultNodeSelector); ok {
		wants.SetDefaultNodeSelector(i.defaultNodeSelector)
	}
}
