package kubectl

import (
	internalclientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	externalclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	core "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
	extensions "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/v1beta1"
)

func versionedCliensetForDeployment(internalClient internalclientset.Interface) externalclientset.Interface {
	if internalClient == nil {
		return &externalclientset.Clientset{}
	}
	return &externalclientset.Clientset{
		CoreV1Client:            core.New(internalClient.Core().RESTClient()),
		ExtensionsV1beta1Client: extensions.New(internalClient.Extensions().RESTClient()),
	}
}
