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

	"k8s.io/apimachinery/pkg/types"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/kubelet/util/csr"
)

// CSRContextAndUser defines the context to use for the client certs in the kubelet kubeconfig file
const CSRContextAndUser = "kubelet-csr"

// PerformTLSBootstrap executes a node certificate signing request.
func PerformTLSBootstrap(cfg *clientcmdapi.Config, hostName string) error {
	client, err := kubeconfigutil.ToClientSet(cfg)
	if err != nil {
		return err
	}

	fmt.Println("[csr] Created API client to obtain unique certificate for this node, generating keys and certificate signing request")

	key, err := certutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		return fmt.Errorf("failed to generate private key [%v]", err)
	}

	cert, err := csr.RequestNodeCertificate(client.CertificatesV1beta1().CertificateSigningRequests(), key, types.NodeName(hostName))
	if err != nil {
		return fmt.Errorf("failed to request signed certificate from the API server [%v]", err)
	}
	fmt.Println("[csr] Received signed certificate from the API server, generating KubeConfig...")

	cfg.AuthInfos[CSRContextAndUser] = &clientcmdapi.AuthInfo{
		ClientKeyData:         key,
		ClientCertificateData: cert,
	}
	cfg.Contexts[CSRContextAndUser] = &clientcmdapi.Context{
		AuthInfo: CSRContextAndUser,
		Cluster:  cfg.Contexts[cfg.CurrentContext].Cluster,
	}
	cfg.CurrentContext = CSRContextAndUser
	return nil
}
