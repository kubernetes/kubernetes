package app

import (
	"k8s.io/apiserver/pkg/admission"
	genericapiserver "k8s.io/apiserver/pkg/server"
	clientgoinformers "k8s.io/client-go/informers"
	"k8s.io/kubernetes/openshift-kube-apiserver/openshiftkubeapiserver"
	"k8s.io/kubernetes/pkg/master"
)

var OpenShiftKubeAPIServerConfigPatch openshiftkubeapiserver.KubeAPIServerConfigFunc = nil

type KubeAPIServerServerFunc func(server *master.Master) error

func PatchKubeAPIServerConfig(config *genericapiserver.Config, versionedInformers clientgoinformers.SharedInformerFactory, pluginInitializers *[]admission.PluginInitializer) (genericapiserver.DelegationTarget, error) {
	if OpenShiftKubeAPIServerConfigPatch == nil {
		return genericapiserver.NewEmptyDelegate(), nil
	}

	return OpenShiftKubeAPIServerConfigPatch(config, versionedInformers, pluginInitializers)
}

var OpenShiftKubeAPIServerServerPatch KubeAPIServerServerFunc = nil

func PatchKubeAPIServerServer(server *master.Master) error {
	if OpenShiftKubeAPIServerServerPatch == nil {
		return nil
	}

	return OpenShiftKubeAPIServerServerPatch(server)
}

var StartingDelegate genericapiserver.DelegationTarget = genericapiserver.NewEmptyDelegate()
