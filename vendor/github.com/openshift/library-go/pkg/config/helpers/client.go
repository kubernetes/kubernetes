package helpers

import (
	"io/ioutil"

	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	configv1 "github.com/openshift/api/config/v1"
	"github.com/openshift/library-go/pkg/config/client"
)

// TODO this file needs to collapse with pkg/config/client.  We cannot safely delegate from this file because this one
// TODO uses JSON and other uses protobuf.

// GetKubeClientConfig loads in-cluster config if kubeConfigFile is empty or the file if not, then applies overrides.
func GetKubeClientConfig(kubeClientConnection configv1.KubeClientConfig) (*rest.Config, error) {
	return GetKubeConfigOrInClusterConfig(kubeClientConnection.KubeConfig, kubeClientConnection.ConnectionOverrides)
}

// GetKubeConfigOrInClusterConfig loads in-cluster config if kubeConfigFile is empty or the file if not,
// then applies overrides.
func GetKubeConfigOrInClusterConfig(kubeConfigFile string, overrides configv1.ClientConnectionOverrides) (*rest.Config, error) {
	if len(kubeConfigFile) > 0 {
		return GetClientConfig(kubeConfigFile, overrides)
	}

	clientConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, err
	}
	applyClientConnectionOverrides(overrides, clientConfig)
	clientConfig.WrapTransport = client.ClientTransportOverrides{WrapTransport: clientConfig.WrapTransport}.DefaultClientTransport

	return clientConfig, nil
}

func GetClientConfig(kubeConfigFile string, overrides configv1.ClientConnectionOverrides) (*rest.Config, error) {
	kubeConfigBytes, err := ioutil.ReadFile(kubeConfigFile)
	if err != nil {
		return nil, err
	}
	kubeConfig, err := clientcmd.NewClientConfigFromBytes(kubeConfigBytes)
	if err != nil {
		return nil, err
	}
	clientConfig, err := kubeConfig.ClientConfig()
	if err != nil {
		return nil, err
	}
	applyClientConnectionOverrides(overrides, clientConfig)
	clientConfig.WrapTransport = client.ClientTransportOverrides{WrapTransport: clientConfig.WrapTransport}.DefaultClientTransport

	return clientConfig, nil
}

// applyClientConnectionOverrides updates a kubeConfig with the overrides from the config.
func applyClientConnectionOverrides(overrides configv1.ClientConnectionOverrides, kubeConfig *rest.Config) {
	if overrides.QPS != 0 {
		kubeConfig.QPS = overrides.QPS
	}
	if overrides.Burst != 0 {
		kubeConfig.Burst = int(overrides.Burst)
	}
	if len(overrides.AcceptContentTypes) != 0 {
		kubeConfig.ContentConfig.AcceptContentTypes = overrides.AcceptContentTypes
	}
	if len(overrides.ContentType) != 0 {
		kubeConfig.ContentConfig.ContentType = overrides.ContentType
	}
}
