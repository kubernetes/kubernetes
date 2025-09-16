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
	"net/url"

	clientset "k8s.io/client-go/kubernetes"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/file"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/https"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/token"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// TokenUser defines token user
const TokenUser = "tls-bootstrap-token-user"

// For returns a kubeconfig object that can be used for doing the TLS Bootstrap with the right credentials
// Also, before returning anything, it makes sure it can trust the API Server
func For(client clientset.Interface, cfg *kubeadmapi.JoinConfiguration) (*clientcmdapi.Config, error) {
	// TODO: Print summary info about the CA certificate, along with the checksum signature
	// we also need an ability for the user to configure the client to validate received CA cert against a checksum
	config, err := DiscoverValidatedKubeConfig(client, cfg)
	if err != nil {
		return nil, errors.Wrap(err, "couldn't validate the identity of the API Server")
	}

	// If the users has provided a TLSBootstrapToken use it for the join process.
	// This is usually the case of Token discovery, but it can also be used with a discovery file
	// without embedded authentication credentials.
	if len(cfg.Discovery.TLSBootstrapToken) != 0 {
		klog.V(1).Info("[discovery] Using provided TLSBootstrapToken as authentication credentials for the join process")

		_, clusterinfo := kubeconfigutil.GetClusterFromKubeConfig(config)
		return kubeconfigutil.CreateWithToken(
			clusterinfo.Server,
			kubeadmapiv1.DefaultClusterName,
			TokenUser,
			clusterinfo.CertificateAuthorityData,
			cfg.Discovery.TLSBootstrapToken,
		), nil
	}

	// if the config returned from discovery has authentication credentials, proceed with the TLS bootstrap process
	if kubeconfigutil.HasAuthenticationCredentials(config) {
		return config, nil
	}

	// if there are no authentication credentials (nor in the config returned from discovery, nor in the TLSBootstrapToken), fail
	return nil, errors.New("couldn't find authentication credentials for the TLS bootstrap process. Please use Token discovery, a discovery file with embedded authentication credentials or a discovery file without authentication credentials but with the TLSBootstrapToken flag")
}

// DiscoverValidatedKubeConfig returns a validated Config object that specifies where the cluster is and the CA cert to trust
func DiscoverValidatedKubeConfig(dryRunClient clientset.Interface, cfg *kubeadmapi.JoinConfiguration) (*clientcmdapi.Config, error) {
	timeout := cfg.Timeouts.Discovery.Duration
	switch {
	case cfg.Discovery.File != nil:
		kubeConfigPath := cfg.Discovery.File.KubeConfigPath
		if isHTTPSURL(kubeConfigPath) {
			return https.RetrieveValidatedConfigInfo(kubeConfigPath, timeout)
		}
		return file.RetrieveValidatedConfigInfo(kubeConfigPath, timeout)
	case cfg.Discovery.BootstrapToken != nil:
		return token.RetrieveValidatedConfigInfo(dryRunClient, &cfg.Discovery, timeout)
	default:
		return nil, errors.New("couldn't find a valid discovery configuration")
	}
}

// isHTTPSURL checks whether the string is parsable as a URL and whether the Scheme is https
func isHTTPSURL(s string) bool {
	u, err := url.Parse(s)
	return err == nil && u.Scheme == "https"
}
