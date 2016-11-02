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

package util

import (
	"fmt"
	"os"
	"path"

	// TODO: "k8s.io/client-go/client/tools/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

func CreateBasicClientConfig(clusterName string, serverURL string, caCert []byte) *clientcmdapi.Config {
	cluster := clientcmdapi.NewCluster()
	cluster.Server = serverURL
	cluster.CertificateAuthorityData = caCert

	config := clientcmdapi.NewConfig()
	config.Clusters[clusterName] = cluster

	return config
}

func MakeClientConfigWithCerts(config *clientcmdapi.Config, clusterName string, userName string, clientKey []byte, clientCert []byte) *clientcmdapi.Config {
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

func MakeClientConfigWithToken(config *clientcmdapi.Config, clusterName string, userName string, token string) *clientcmdapi.Config {
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
