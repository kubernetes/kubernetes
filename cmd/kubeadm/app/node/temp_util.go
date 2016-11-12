/*
Copyright 2016 The Kubernetes Authors.

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


// TODO: this whole file is just a temporary solution until the transition to k8s.io/client-go is finished.
// It contains copies of the functions in kubeadm/app/cmd/util/kubeconfig.go that work with a non-versioned client
// and should be removed in favor of the original kubeconfig.go file once the signature of csr.RequestNodeCertificate() is fixed.

package node

import (
	"os"
	"path"
	"fmt"

	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// TODO: remove this in favor of kubeadmutil.WriteKubeconfigIfNotExists() when csr.RequestNodeCertificate() is fixed
func WriteKubeconfigIfNotExists(name string, kubeconfig *clientcmdapi.Config) error {
	if err := os.MkdirAll(kubeadmapi.GlobalEnvParams.KubernetesDir, 0700); err != nil {
		return fmt.Errorf("<util/kubeconfig> failed to create directory %q [%v]", kubeadmapi.GlobalEnvParams.KubernetesDir, err)
	}

	filename := path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, fmt.Sprintf("%s.conf", name))
	// Create and open the file, only if it does not already exist.
	f, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0600)
	if err != nil {
		return fmt.Errorf("<util/kubeconfig> failed to create %q, it already exists [%v]", filename, err)
	}
	f.Close()

	if err := clientcmd.WriteToFile(*kubeconfig, filename); err != nil {
		return fmt.Errorf("<util/kubeconfig> failed to write to %q [%v]", filename, err)
	}

	fmt.Printf("<util/kubeconfig> created %q\n", filename)
	return nil
}

// TODO: remove this in favor of kubeadmutil.CreateBasicClientConfig() when csr.RequestNodeCertificate() is fixed
func createBasicClientConfig(clusterName string, serverURL string, caCert []byte) *clientcmdapi.Config {
	cluster := clientcmdapi.NewCluster()
	cluster.Server = serverURL
	cluster.CertificateAuthorityData = caCert

	config := clientcmdapi.NewConfig()
	config.Clusters[clusterName] = cluster

	return config
}

// TODO: remove this in favor of kubeadmutil.MakeClientConfigWithToken() when csr.RequestNodeCertificate() is fixed
func makeClientConfigWithToken(config *clientcmdapi.Config, clusterName string, userName string, token string) *clientcmdapi.Config {
	newConfig := config
	name := fmt.Sprintf("%s@%s", userName, clusterName)

	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.Token = token

	context := clientcmdapi.NewContext()
	context.Cluster = clusterName
	context.AuthInfo = userName

	newConfig.AuthInfos[userName] = authInfo
	newConfig.Contexts[name] = context
	newConfig.CurrentContext = name

	return newConfig
}

// TODO: remove this in favor of kubeadmutil.MakeClientConfigWithCerts() when csr.RequestNodeCertificate() is fixed
func makeClientConfigWithCerts(config *clientcmdapi.Config, clusterName string, userName string, clientKey []byte, clientCert []byte) *clientcmdapi.Config {
	newConfig := config
	name := fmt.Sprintf("%s@%s", userName, clusterName)

	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.ClientKeyData = clientKey
	authInfo.ClientCertificateData = clientCert

	context := clientcmdapi.NewContext()
	context.Cluster = clusterName
	context.AuthInfo = userName

	newConfig.AuthInfos[userName] = authInfo
	newConfig.Contexts[name] = context
	newConfig.CurrentContext = name

	return newConfig
}
