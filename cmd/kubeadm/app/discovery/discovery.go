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

package discovery

import (
	"fmt"
	"net/url"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/file"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/https"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/token"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

const TokenUser = "tls-bootstrap-token-user"

// For returns a KubeConfig object that can be used for doing the TLS Bootstrap with the right credentials
// Also, before returning anything, it makes sure it can trust the API Server
func For(cfg *kubeadmapi.NodeConfiguration) (*clientcmdapi.Config, error) {
	// TODO: Print summary info about the CA certificate, along with the the checksum signature
	// we also need an ability for the user to configure the client to validate received CA cert against a checksum
	clusterinfo, err := GetValidatedClusterInfoObject(cfg)
	if err != nil {
		return nil, fmt.Errorf("couldn't validate the identity of the API Server: %v", err)
	}

	return kubeconfigutil.CreateWithToken(
		clusterinfo.Server,
		"kubernetes",
		TokenUser,
		clusterinfo.CertificateAuthorityData,
		cfg.TLSBootstrapToken,
	), nil
}

// GetValidatedClusterInfoObject returns a validated Cluster object that specifies where the cluster is and the CA cert to trust
func GetValidatedClusterInfoObject(cfg *kubeadmapi.NodeConfiguration) (*clientcmdapi.Cluster, error) {
	switch {
	case len(cfg.DiscoveryFile) != 0:
		if isHTTPSURL(cfg.DiscoveryFile) {
			return https.RetrieveValidatedClusterInfo(cfg.DiscoveryFile)
		}
		return file.RetrieveValidatedClusterInfo(cfg.DiscoveryFile)
	case len(cfg.DiscoveryToken) != 0:
		return token.RetrieveValidatedClusterInfo(cfg.DiscoveryToken, cfg.DiscoveryTokenAPIServers)
	default:
		return nil, fmt.Errorf("couldn't find a valid discovery configuration.")
	}
}

// isHTTPSURL checks whether the string is parsable as an URL
func isHTTPSURL(s string) bool {
	u, err := url.Parse(s)
	return err == nil && u.Scheme == "https"
}
