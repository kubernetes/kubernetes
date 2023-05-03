package imagepolicy

import (
	"k8s.io/apiserver/pkg/admission"

	"github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/imagereferencemutators"
)

func NewInitializer(imageMutators imagereferencemutators.ImageMutators, internalImageRegistry string) admission.PluginInitializer {
	return &localInitializer{
		imageMutators:         imageMutators,
		internalImageRegistry: internalImageRegistry,
	}
}

type WantsImageMutators interface {
	SetImageMutators(imagereferencemutators.ImageMutators)
	admission.InitializationValidator
}

type WantsInternalImageRegistry interface {
	SetInternalImageRegistry(string)
	admission.InitializationValidator
}

type localInitializer struct {
	imageMutators         imagereferencemutators.ImageMutators
	internalImageRegistry string
}

// Initialize will check the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *localInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsImageMutators); ok {
		wants.SetImageMutators(i.imageMutators)
	}
	if wants, ok := plugin.(WantsInternalImageRegistry); ok {
		wants.SetInternalImageRegistry(i.internalImageRegistry)
	}
}
