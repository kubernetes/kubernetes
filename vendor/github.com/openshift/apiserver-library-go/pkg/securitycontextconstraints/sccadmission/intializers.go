package sccadmission

import (
	"k8s.io/apiserver/pkg/admission"

	securityv1informer "github.com/openshift/client-go/security/informers/externalversions/security/v1"
)

func NewInitializer(sccInformer securityv1informer.SecurityContextConstraintsInformer) admission.PluginInitializer {
	return &localInitializer{sccInformer: sccInformer}
}

type WantsSecurityInformer interface {
	SetSecurityInformers(securityv1informer.SecurityContextConstraintsInformer)
	admission.InitializationValidator
}

type localInitializer struct {
	sccInformer securityv1informer.SecurityContextConstraintsInformer
}

// Initialize will check the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *localInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsSecurityInformer); ok {
		wants.SetSecurityInformers(i.sccInformer)
	}
}
