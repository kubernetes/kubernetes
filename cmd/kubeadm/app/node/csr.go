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

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubelet/util/csr"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

// PerformTLSBootstrap executes a node certificate signing request.
func PerformTLSBootstrap(cfg *clientcmdapi.Config) error {
	hostName, err := os.Hostname()
	if err != nil {
		return err
	}
	name := types.NodeName(hostName)

	rc, err := clientcmd.NewDefaultClientConfig(*cfg, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return err
	}
	c, err := clientset.NewForConfig(rc)
	if err != nil {
		return err
	}
	fmt.Println("[csr] Created API client to obtain unique certificate for this node, generating keys and certificate signing request")

	key, err := certutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		return fmt.Errorf("failed to generate private key [%v]", err)
	}

	// Make sure there are no other nodes in the cluster with identical node name.
	if err := checkForNodeNameDuplicates(c); err != nil {
		return err
	}

	cert, err := csr.RequestNodeCertificate(c.Certificates().CertificateSigningRequests(), key, name)
	if err != nil {
		return fmt.Errorf("failed to request signed certificate from the API server [%v]", err)
	}
	fmt.Printf("[csr] Received signed certificate from the API server")
	fmt.Println("[csr] Generating kubelet configuration")

	cfg.AuthInfos["kubelet"] = &clientcmdapi.AuthInfo{
		ClientKeyData:         key,
		ClientCertificateData: cert,
	}
	cfg.Contexts["kubelet"] = &clientcmdapi.Context{
		AuthInfo: "kubelet",
		Cluster:  cfg.Contexts[cfg.CurrentContext].Cluster,
	}
	cfg.CurrentContext = "kubelet"
	return nil
}
