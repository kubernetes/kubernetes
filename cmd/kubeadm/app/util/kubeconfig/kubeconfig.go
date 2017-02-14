/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kubeconfig

import (
	"fmt"
	"os"
	"path/filepath"
	"io/ioutil"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

const (
	KubernetesDirPermissions    = 0700
	KubeConfigFilePermissions   = 0600
)

func CreateBasicClientConfig(serverURL string, clusterName string, userName string, caCert []byte) *clientcmdapi.Config {
	config := clientcmdapi.NewConfig()

	// Make a new cluster, specify the endpoint we'd like to talk to and the ca cert we're gonna use
	cluster := clientcmdapi.NewCluster()
	cluster.Server = serverURL
	cluster.CertificateAuthorityData = caCert

	// Specify a context where we're using that cluster and the username as the auth information
	contextName := fmt.Sprintf("%s@%s", userName, clusterName)
	context := clientcmdapi.NewContext()
	context.Cluster = clusterName
	context.AuthInfo = userName

	// Lastly, apply the created objects above to the config
	config.Clusters[clusterName] = cluster
	config.Contexts[contextName] = context
	config.CurrentContext = contextName
	return config
}

// Creates a clientcmdapi.Config object with access to the API server with client certificates
func MakeClientConfigWithCerts(serverURL, clusterName, userName string, caCert []byte, clientKey []byte, clientCert []byte) *clientcmdapi.Config {
	config := CreateBasicClientConfig(serverURL, clusterName, userName, caCert)

	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.ClientKeyData = clientKey
	authInfo.ClientCertificateData = clientCert

	config.AuthInfos[userName] = authInfo
	return config
}

// Creates a clientcmdapi.Config object with access to the API server with a token
func MakeClientConfigWithToken(serverURL, clusterName, userName string, caCert []byte, token string) *clientcmdapi.Config {
	config := CreateBasicClientConfig(serverURL, clusterName, userName, caCert)

	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.Token = token

	config.AuthInfos[userName] = authInfo
	return config
}

func CreateClientFromFile(path string) (*clientset.Clientset, error) {
	config, err := clientcmd.LoadFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load admin kubeconfig [%v]", err)
	}
	return KubeConfigToClientSet(config)
}

func KubeConfigToClientSet(config *clientcmdapi.Config) (*clientset.Clientset, error) {
	clientConfig, err := clientcmd.NewDefaultClientConfig(*config, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to create API client configuration from kubeconfig: %v", err)
	}

	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create API client [%v]", err)
	}
	return client, nil
}

func WriteKubeconfigToDisk(filename string, kubeconfig *clientcmdapi.Config) error {
	// Convert the KubeConfig object to a byte array
	content, err := clientcmd.Write(*kubeconfig)
	if err != nil {
		return err
	}

	// Create the directory if it does not exist
	dir := filepath.Dir(filename)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		if err = os.MkdirAll(dir, KubernetesDirPermissions); err != nil {
			return err
		}
	}

	// No such kubeconfig file exists; write that kubeconfig down to disk then
	if err := ioutil.WriteFile(filename, content, KubeConfigFilePermissions); err != nil {
		return err
	}

	fmt.Printf("[kubeconfig] Wrote KubeConfig file to disk: %q\n", filename)
	return nil
}
