/*
Copyright 2018 The Kubernetes Authors.

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
	"bytes"
	"crypto/x509"
	"html/template"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/client-go/tools/clientcmd"
	clientcertutil "k8s.io/client-go/util/cert"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pubkeypin"
)

var joinCommandTemplate = template.Must(template.New("join").Parse(`` +
	`kubeadm join {{.MasterHostPort}} --token {{.Token}}{{range $h := .CAPubKeyPins}} --discovery-token-ca-cert-hash {{$h}}{{end}}`,
))

// GetJoinCommand returns the kubeadm join command for a given token and
// and Kubernetes cluster (the current cluster in the kubeconfig file)
func GetJoinCommand(kubeConfigFile string, token string, skipTokenPrint bool) (string, error) {
	// load the kubeconfig file to get the CA certificate and endpoint
	config, err := clientcmd.LoadFromFile(kubeConfigFile)
	if err != nil {
		return "", errors.Wrap(err, "failed to load kubeconfig")
	}

	// load the default cluster config
	clusterConfig := kubeconfigutil.GetClusterFromKubeConfig(config)
	if clusterConfig == nil {
		return "", errors.New("failed to get default cluster config")
	}

	// load CA certificates from the kubeconfig (either from PEM data or by file path)
	var caCerts []*x509.Certificate
	if clusterConfig.CertificateAuthorityData != nil {
		caCerts, err = clientcertutil.ParseCertsPEM(clusterConfig.CertificateAuthorityData)
		if err != nil {
			return "", errors.Wrap(err, "failed to parse CA certificate from kubeconfig")
		}
	} else if clusterConfig.CertificateAuthority != "" {
		caCerts, err = clientcertutil.CertsFromFile(clusterConfig.CertificateAuthority)
		if err != nil {
			return "", errors.Wrap(err, "failed to load CA certificate referenced by kubeconfig")
		}
	} else {
		return "", errors.New("no CA certificates found in kubeconfig")
	}

	// hash all the CA certs and include their public key pins as trusted values
	publicKeyPins := make([]string, 0, len(caCerts))
	for _, caCert := range caCerts {
		publicKeyPins = append(publicKeyPins, pubkeypin.Hash(caCert))
	}

	ctx := map[string]interface{}{
		"Token":          token,
		"CAPubKeyPins":   publicKeyPins,
		"MasterHostPort": strings.Replace(clusterConfig.Server, "https://", "", -1),
	}

	if skipTokenPrint {
		ctx["Token"] = template.HTML("<value withheld>")
	}

	var out bytes.Buffer
	err = joinCommandTemplate.Execute(&out, ctx)
	if err != nil {
		return "", errors.Wrap(err, "failed to render join command template")
	}
	return out.String(), nil
}
