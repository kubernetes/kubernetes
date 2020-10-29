package restrictusers

import (
	"k8s.io/apiserver/pkg/admission"

	userinformer "github.com/openshift/client-go/user/informers/externalversions"
)

func NewInitializer(userInformer userinformer.SharedInformerFactory) admission.PluginInitializer {
	return &localInitializer{userInformer: userInformer}
}

type WantsUserInformer interface {
	SetUserInformer(userinformer.SharedInformerFactory)
	admission.InitializationValidator
}

type localInitializer struct {
	userInformer userinformer.SharedInformerFactory
}

// Initialize will check the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *localInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsUserInformer); ok {
		wants.SetUserInformer(i.userInformer)
	}
}
