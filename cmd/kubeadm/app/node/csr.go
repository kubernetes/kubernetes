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

package node

import (
	"fmt"
	"os"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/api"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/apis/certificates"
	unversionedcertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubelet/util/csr"
	"k8s.io/kubernetes/pkg/types"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

// PerformTLSBootstrap creates a RESTful client in order to execute certificate signing request.
func PerformTLSBootstrap(s *kubeadmapi.NodeConfiguration, apiEndpoint string, caCert []byte) (*clientcmdapi.Config, error) {
	// TODO(phase1+) try all the api servers until we find one that works
	bareClientConfig := kubeadmutil.CreateBasicClientConfig("kubernetes", apiEndpoint, caCert)

	hostName, err := os.Hostname()
	if err != nil {
		return nil, fmt.Errorf("<node/csr> failed to get node hostname [%v]", err)
	}

	// TODO(phase1+) https://github.com/kubernetes/kubernetes/issues/33641
	nodeName := types.NodeName(hostName)

	bootstrapClientConfig, err := clientcmd.NewDefaultClientConfig(
		*kubeadmutil.MakeClientConfigWithToken(
			bareClientConfig, "kubernetes", fmt.Sprintf("kubelet-%s", nodeName), s.Secrets.BearerToken,
		),
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("<node/csr> failed to create API client configuration [%v]", err)
	}

	client, err := unversionedcertificates.NewForConfig(bootstrapClientConfig)
	if err != nil {
		return nil, fmt.Errorf("<node/csr> failed to create API client [%v]", err)
	}
	csrClient := client.CertificateSigningRequests()

	// TODO(phase1+) https://github.com/kubernetes/kubernetes/issues/33643

	if err := checkCertsAPI(bootstrapClientConfig); err != nil {
		return nil, fmt.Errorf("<node/csr> failed to proceed due to API compatibility issue - %v", err)
	}

	fmt.Println("<node/csr> created API client to obtain unique certificate for this node, generating keys and certificate signing request")

	key, err := certutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		return nil, fmt.Errorf("<node/csr> failed to generating private key [%v]", err)
	}
	cert, err := csr.RequestNodeCertificate(csrClient, key, nodeName)
	if err != nil {
		return nil, fmt.Errorf("<node/csr> failed to request signed certificate from the API server [%v]", err)
	}
	fmtCert, err := certutil.FormatBytesCert(cert)
	if err != nil {
		return nil, fmt.Errorf("<node/csr> failed to format certificate [%v]", err)
	}
	fmt.Printf("<node/csr> received signed certificate from the API server:\n%s\n", fmtCert)
	fmt.Println("<node/csr> generating kubelet configuration")
	finalConfig := kubeadmutil.MakeClientConfigWithCerts(
		bareClientConfig, "kubernetes", fmt.Sprintf("kubelet-%s", nodeName),
		key, cert,
	)

	return finalConfig, nil
}

func checkCertsAPI(config *restclient.Config) error {
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)

	if err != nil {
		return fmt.Errorf("failed to create API discovery client [%v]", err)
	}

	serverGroups, err := discoveryClient.ServerGroups()

	if err != nil {
		return fmt.Errorf("failed to retrieve a list of supported API objects [%v]", err)
	}

	for _, group := range serverGroups.Groups {
		if group.Name == certificates.GroupName {
			return nil
		}
	}

	version, err := discoveryClient.ServerVersion()
	if err != nil {
		return fmt.Errorf("unable to obtain API version [%v]", err)
	}

	return fmt.Errorf("API version %s does not support certificates API, use v1.4.0 or newer", version.String())
}
