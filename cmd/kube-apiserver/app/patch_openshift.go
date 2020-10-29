package app

import (
	"k8s.io/apiserver/pkg/admission"
	genericapiserver "k8s.io/apiserver/pkg/server"
	clientgoinformers "k8s.io/client-go/informers"
	"k8s.io/kubernetes/openshift-kube-apiserver/openshiftkubeapiserver"
)

var OpenShiftKubeAPIServerConfigPatch openshiftkubeapiserver.KubeAPIServerConfigFunc = nil

func PatchKubeAPIServerConfig(config *genericapiserver.Config, versionedInformers clientgoinformers.SharedInformerFactory, pluginInitializers *[]admission.PluginInitializer) error {
	if OpenShiftKubeAPIServerConfigPatch == nil {
		return nil
	}

	return OpenShiftKubeAPIServerConfigPatch(config, versionedInformers, pluginInitializers)
}
