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

package kubemaster

import (
	"crypto/rsa"
	"crypto/x509"
	"fmt"

	// TODO: "k8s.io/client-go/client/tools/clientcmd/api"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

func CreateCertsAndConfigForClients(params *kubeadmapi.BootstrapParams, clientNames []string, caKey *rsa.PrivateKey, caCert *x509.Certificate) (map[string]*clientcmdapi.Config, error) {

	basicClientConfig := kubeadmutil.CreateBasicClientConfig(
		"kubernetes",
		fmt.Sprintf("https://%s:443", params.Discovery.ListenIP),
		certutil.EncodeCertPEM(caCert),
	)

	configs := map[string]*clientcmdapi.Config{}

	for _, client := range clientNames {
		key, cert, err := newClientKeyAndCert(caCert, caKey)
		if err != nil {
			return nil, fmt.Errorf("<master/kubeconfig> failure while creating %s client certificate - %s", client, err)
		}
		config := kubeadmutil.MakeClientConfigWithCerts(
			basicClientConfig,
			"kubernetes",
			client,
			certutil.EncodePrivateKeyPEM(key),
			certutil.EncodeCertPEM(cert),
		)
		configs[client] = config
	}

	return configs, nil
}
