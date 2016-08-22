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

package kubenode

import (
	"fmt"
	"io/ioutil"
	"strings"

	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	unversionedcertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"
	utilcertificates "k8s.io/kubernetes/pkg/util/certificates"
)

func getNodeName() string {
	return "TODO"
}

// runs on nodes
func PerformTLSBootstrap(params *kubeadmapi.BootstrapParams) (*clientcmdapi.Config, error) {
	// Create a restful client for doing the certificate signing request.
	pemData, err := ioutil.ReadFile(params.Discovery.CaCertFile)
	if err != nil {
		return nil, err
	}
	// TODO try all the api servers until we find one that works
	configBootstrap := kubeadmutil.CreateBasicClientConfig(
		"kubernetes", strings.Split(params.Discovery.ApiServerURLs, ",")[0], pemData,
	)

	nodeName := getNodeName()

	configBootstrap = kubeadmutil.MakeClientConfigWithToken(
		configBootstrap, "kubernetes", fmt.Sprintf("kubelet-%s", nodeName), params.Discovery.BearerToken,
	)
	clientConfig, err := clientcmd.NewDefaultClientConfig(
		*configBootstrap,
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, err
	}

	client, err := unversionedcertificates.NewForConfig(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("unable to create certificates signing request client: %v", err)
	}
	csrClient := client.CertificateSigningRequests()

	keyData, err := utilcertificates.GeneratePrivateKey()
	if err != nil {
		return nil, fmt.Errorf("error generating key: %v", err)
	}
	// Pass 'requestClientCertificate()' the CSR client, existing key data, and node name to
	// request for client certificate from the API server.
	certData, err := kubeletapp.RequestClientCertificate(csrClient, keyData, nodeName)
	if err != nil {
		return nil, fmt.Errorf("unable to request certificate from API server: %v", err)
	}
	// TODO transform clientcert into kubeconfig so that it can be written out on the node

	finalConfig := kubeadmutil.MakeClientConfigWithCerts(
		configBootstrap, "kubernetes", fmt.Sprintf("kubelet-%s", nodeName),
		keyData, certData,
	)

	return finalConfig, nil
}
