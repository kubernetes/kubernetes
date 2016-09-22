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
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
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

// kubeadm is responsible for writing the following kubeconfig file, which
// kubelet should be waiting for. Help user avoid foot-shooting by refusing to
// write a file that has already been written (the kubelet will be up and
// running in that case - they'd need to stop the kubelet, remove the file, and
// start it again in that case).

func WriteKubeconfigIfNotExists(s *kubeadmapi.KubeadmConfig, name string, kubeconfig *clientcmdapi.Config) error {
	if err := os.MkdirAll(s.EnvParams["prefix"], 0700); err != nil {
		return fmt.Errorf("<util/kubeconfig> failed to create directory %q [%s]", s.EnvParams["prefix"], err)
	}

	filename := path.Join(s.EnvParams["prefix"], fmt.Sprintf("%s.conf", name))
	// Create and open the file, only if it does not already exist.
	f, err := os.OpenFile(
		filename,
		os.O_CREATE|os.O_WRONLY|os.O_EXCL,
		0600,
	)
	if err != nil {
		return fmt.Errorf("<util/kubeconfig> failed to create %q, it already exists [%s]", filename, err)
	}
	f.Close()

	if err := clientcmd.WriteToFile(*kubeconfig, filename); err != nil {
		return fmt.Errorf("<util/kubeconfig> failed to write to %q [%s]", filename, err)
	}

	fmt.Printf("<util/kubeconfig> created %q\n", filename)
	return nil
}
