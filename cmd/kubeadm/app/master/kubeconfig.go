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

package master

import (
	"crypto/rsa"
	"crypto/x509"
	"fmt"

	// TODO: "k8s.io/client-go/client/tools/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

func CreateCertsAndConfigForClients(cfg kubeadmapi.API, clientNames []string, caKey *rsa.PrivateKey, caCert *x509.Certificate) (map[string]*clientcmdapi.Config, error) {

	basicClientConfig := kubeadmutil.CreateBasicClientConfig(
		"kubernetes",
		// TODO this is not great, but there is only one address we can use here
		// so we'll pick the first one, there is much of chance to have an empty
		// slice by the time this gets called
		fmt.Sprintf("https://%s:%d", cfg.AdvertiseAddresses[0], cfg.BindPort),
		certutil.EncodeCertPEM(caCert),
	)

	configs := map[string]*clientcmdapi.Config{}

	for _, client := range clientNames {
		key, cert, err := newClientKeyAndCert(caCert, caKey)
		if err != nil {
			return nil, fmt.Errorf("<master/kubeconfig> failure while creating %s client certificate - %v", client, err)
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
