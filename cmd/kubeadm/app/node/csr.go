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

	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubelet/util/csr"
	"k8s.io/kubernetes/pkg/types"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

func PerformTLSBootstrapDeprecated(connection *ConnectionDetails) (*clientcmdapi.Config, error) {
	csrClient := connection.CertClient.CertificateSigningRequests()

	fmt.Println("[csr] Created API client to obtain unique certificate for this node, generating keys and certificate signing request")

	key, err := certutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		return nil, fmt.Errorf("failed to generating private key [%v]", err)
	}
	cert, err := csr.RequestNodeCertificate(csrClient, key, connection.NodeName)
	if err != nil {
		return nil, fmt.Errorf("failed to request signed certificate from the API server [%v]", err)
	}
	fmtCert, err := certutil.FormatBytesCert(cert)
	if err != nil {
		return nil, fmt.Errorf("failed to format certificate [%v]", err)
	}
	fmt.Printf("[csr] Received signed certificate from the API server:\n%s\n", fmtCert)
	fmt.Println("[csr] Generating kubelet configuration")

	bareClientConfig := kubeadmutil.CreateBasicClientConfig("kubernetes", connection.Endpoint, connection.CACert)
	finalConfig := kubeadmutil.MakeClientConfigWithCerts(
		bareClientConfig, "kubernetes", fmt.Sprintf("kubelet-%s", connection.NodeName),
		key, cert,
	)

	return finalConfig, nil
}

// PerformTLSBootstrap executes a certificate signing request with the
// provided connection details.
func PerformTLSBootstrap(cfg *clientcmdapi.Config) error {
	hostName, err := os.Hostname()
	if err != nil {
		return err
	}
	name := types.NodeName(hostName)

	rc, err := clientcmd.NewDefaultClientConfig(*cfg, nil).ClientConfig()
	if err != nil {
		return err
	}
	c, err := clientset.NewForConfig(rc)
	if err != nil {
		return err
	}
	fmt.Println("<node/csr> created API client to obtain unique certificate for this node, generating keys and certificate signing request")

	key, err := certutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		return fmt.Errorf("<node/csr> failed to generating private key [%v]", err)
	}
	cert, err := csr.RequestNodeCertificate(c.Certificates().CertificateSigningRequests(), key, name)
	if err != nil {
		return fmt.Errorf("<node/csr> failed to request signed certificate from the API server [%v]", err)
	}
	fmtCert, err := certutil.FormatBytesCert(cert)
	if err != nil {
		return fmt.Errorf("<node/csr> failed to format certificate [%v]", err)
	}
	fmt.Printf("<node/csr> received signed certificate from the API server:\n%s\n", fmtCert)
	fmt.Println("<node/csr> generating kubelet configuration")

	cfg.AuthInfos["kubelet"] = &clientcmdapi.AuthInfo{
		ClientKeyData:         key,
		ClientCertificateData: []byte(fmtCert),
	}
	cfg.Contexts["kubelet"] = &clientcmdapi.Context{
		AuthInfo: "kubelet",
		Cluster:  cfg.Contexts[cfg.CurrentContext].Cluster,
	}
	cfg.CurrentContext = "kubelet"
	return nil
}
