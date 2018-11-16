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

package token

import (
	"bytes"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"sync"
	"time"

	"github.com/pkg/errors"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pubkeypin"
	"k8s.io/kubernetes/pkg/controller/bootstrap"
)

// BootstrapUser defines bootstrap user name
const BootstrapUser = "token-bootstrap-client"

// RetrieveValidatedConfigInfo connects to the API Server and tries to fetch the cluster-info ConfigMap
// It then makes sure it can trust the API Server by looking at the JWS-signed tokens and (if cfg.DiscoveryTokenCACertHashes is not empty)
// validating the cluster CA against a set of pinned public keys
func RetrieveValidatedConfigInfo(cfg *kubeadmapi.JoinConfiguration) (*clientcmdapi.Config, error) {
	token, err := kubeadmapi.NewBootstrapTokenString(cfg.Discovery.BootstrapToken.Token)
	if err != nil {
		return nil, err
	}

	// Load the cfg.DiscoveryTokenCACertHashes into a pubkeypin.Set
	pubKeyPins := pubkeypin.NewSet()
	err = pubKeyPins.Allow(cfg.Discovery.BootstrapToken.CACertHashes...)
	if err != nil {
		return nil, err
	}

	// The function below runs for every endpoint, and all endpoints races with each other.
	// The endpoint that wins the race and completes the task first gets its kubeconfig returned below
	baseKubeConfig, err := fetchKubeConfigWithTimeout(cfg.Discovery.BootstrapToken.APIServerEndpoint, cfg.Discovery.Timeout.Duration, func(endpoint string) (*clientcmdapi.Config, error) {

		insecureBootstrapConfig := buildInsecureBootstrapKubeConfig(endpoint, kubeadmapiv1beta1.DefaultClusterName)
		clusterName := insecureBootstrapConfig.Contexts[insecureBootstrapConfig.CurrentContext].Cluster

		insecureClient, err := kubeconfigutil.ToClientSet(insecureBootstrapConfig)
		if err != nil {
			return nil, err
		}

		fmt.Printf("[discovery] Created cluster-info discovery client, requesting info from %q\n", insecureBootstrapConfig.Clusters[clusterName].Server)

		// Make an initial insecure connection to get the cluster-info ConfigMap
		var insecureClusterInfo *v1.ConfigMap
		wait.PollImmediateInfinite(constants.DiscoveryRetryInterval, func() (bool, error) {
			var err error
			insecureClusterInfo, err = insecureClient.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
			if err != nil {
				fmt.Printf("[discovery] Failed to request cluster info, will try again: [%s]\n", err)
				return false, nil
			}
			return true, nil
		})

		// Validate the MAC on the kubeconfig from the ConfigMap and load it
		insecureKubeconfigString, ok := insecureClusterInfo.Data[bootstrapapi.KubeConfigKey]
		if !ok || len(insecureKubeconfigString) == 0 {
			return nil, errors.Errorf("there is no %s key in the %s ConfigMap. This API Server isn't set up for token bootstrapping, can't connect",
				bootstrapapi.KubeConfigKey, bootstrapapi.ConfigMapClusterInfo)
		}
		detachedJWSToken, ok := insecureClusterInfo.Data[bootstrapapi.JWSSignatureKeyPrefix+token.ID]
		if !ok || len(detachedJWSToken) == 0 {
			return nil, errors.Errorf("token id %q is invalid for this cluster or it has expired. Use \"kubeadm token create\" on the master node to creating a new valid token", token.ID)
		}
		if !bootstrap.DetachedTokenIsValid(detachedJWSToken, insecureKubeconfigString, token.ID, token.Secret) {
			return nil, errors.New("failed to verify JWS signature of received cluster info object, can't trust this API Server")
		}
		insecureKubeconfigBytes := []byte(insecureKubeconfigString)
		insecureConfig, err := clientcmd.Load(insecureKubeconfigBytes)
		if err != nil {
			return nil, errors.Wrapf(err, "couldn't parse the kubeconfig file in the %s configmap", bootstrapapi.ConfigMapClusterInfo)
		}

		// If no TLS root CA pinning was specified, we're done
		if pubKeyPins.Empty() {
			fmt.Printf("[discovery] Cluster info signature and contents are valid and no TLS pinning was specified, will use API Server %q\n", endpoint)
			return insecureConfig, nil
		}

		// Load the cluster CA from the Config
		if len(insecureConfig.Clusters) != 1 {
			return nil, errors.Errorf("expected the kubeconfig file in the %s configmap to have a single cluster, but it had %d", bootstrapapi.ConfigMapClusterInfo, len(insecureConfig.Clusters))
		}
		var clusterCABytes []byte
		for _, cluster := range insecureConfig.Clusters {
			clusterCABytes = cluster.CertificateAuthorityData
		}
		clusterCA, err := parsePEMCert(clusterCABytes)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to parse cluster CA from the %s configmap", bootstrapapi.ConfigMapClusterInfo)

		}

		// Validate the cluster CA public key against the pinned set
		err = pubKeyPins.Check(clusterCA)
		if err != nil {
			return nil, errors.Wrapf(err, "cluster CA found in %s configmap is invalid", bootstrapapi.ConfigMapClusterInfo)
		}

		// Now that we know the proported cluster CA, connect back a second time validating with that CA
		secureBootstrapConfig := buildSecureBootstrapKubeConfig(endpoint, clusterCABytes, clusterName)
		secureClient, err := kubeconfigutil.ToClientSet(secureBootstrapConfig)
		if err != nil {
			return nil, err
		}

		fmt.Printf("[discovery] Requesting info from %q again to validate TLS against the pinned public key\n", insecureBootstrapConfig.Clusters[clusterName].Server)
		var secureClusterInfo *v1.ConfigMap
		wait.PollImmediateInfinite(constants.DiscoveryRetryInterval, func() (bool, error) {
			var err error
			secureClusterInfo, err = secureClient.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
			if err != nil {
				fmt.Printf("[discovery] Failed to request cluster info, will try again: [%s]\n", err)
				return false, nil
			}
			return true, nil
		})

		// Pull the kubeconfig from the securely-obtained ConfigMap and validate that it's the same as what we found the first time
		secureKubeconfigBytes := []byte(secureClusterInfo.Data[bootstrapapi.KubeConfigKey])
		if !bytes.Equal(secureKubeconfigBytes, insecureKubeconfigBytes) {
			return nil, errors.Errorf("the second kubeconfig from the %s configmap (using validated TLS) was different from the first", bootstrapapi.ConfigMapClusterInfo)
		}

		secureKubeconfig, err := clientcmd.Load(secureKubeconfigBytes)
		if err != nil {
			return nil, errors.Wrapf(err, "couldn't parse the kubeconfig file in the %s configmap", bootstrapapi.ConfigMapClusterInfo)
		}

		fmt.Printf("[discovery] Cluster info signature and contents are valid and TLS certificate validates against pinned roots, will use API Server %q\n", endpoint)
		return secureKubeconfig, nil
	})
	if err != nil {
		return nil, err
	}

	return baseKubeConfig, nil
}

// buildInsecureBootstrapKubeConfig makes a kubeconfig object that connects insecurely to the API Server for bootstrapping purposes
func buildInsecureBootstrapKubeConfig(endpoint, clustername string) *clientcmdapi.Config {
	masterEndpoint := fmt.Sprintf("https://%s", endpoint)
	bootstrapConfig := kubeconfigutil.CreateBasic(masterEndpoint, clustername, BootstrapUser, []byte{})
	bootstrapConfig.Clusters[clustername].InsecureSkipTLSVerify = true
	return bootstrapConfig
}

// buildSecureBootstrapKubeConfig makes a kubeconfig object that connects securely to the API Server for bootstrapping purposes (validating with the specified CA)
func buildSecureBootstrapKubeConfig(endpoint string, caCert []byte, clustername string) *clientcmdapi.Config {
	masterEndpoint := fmt.Sprintf("https://%s", endpoint)
	bootstrapConfig := kubeconfigutil.CreateBasic(masterEndpoint, clustername, BootstrapUser, caCert)
	return bootstrapConfig
}

// fetchKubeConfigWithTimeout tries to run fetchKubeConfigFunc on every DiscoveryRetryInterval, but until discoveryTimeout is reached
func fetchKubeConfigWithTimeout(apiEndpoint string, discoveryTimeout time.Duration, fetchKubeConfigFunc func(string) (*clientcmdapi.Config, error)) (*clientcmdapi.Config, error) {
	stopChan := make(chan struct{})
	var resultingKubeConfig *clientcmdapi.Config
	var once sync.Once
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		wait.Until(func() {
			fmt.Printf("[discovery] Trying to connect to API Server %q\n", apiEndpoint)
			cfg, err := fetchKubeConfigFunc(apiEndpoint)
			if err != nil {
				fmt.Printf("[discovery] Failed to connect to API Server %q: %v\n", apiEndpoint, err)
				return
			}
			fmt.Printf("[discovery] Successfully established connection with API Server %q\n", apiEndpoint)
			once.Do(func() {
				resultingKubeConfig = cfg
				close(stopChan)
			})
		}, constants.DiscoveryRetryInterval, stopChan)
	}()

	select {
	case <-time.After(discoveryTimeout):
		once.Do(func() {
			close(stopChan)
		})
		err := errors.Errorf("abort connecting to API servers after timeout of %v", discoveryTimeout)
		fmt.Printf("[discovery] %v\n", err)
		wg.Wait()
		return nil, err
	case <-stopChan:
		wg.Wait()
		return resultingKubeConfig, nil
	}
}

// parsePEMCert decodes a PEM-formatted certificate and returns it as an x509.Certificate
func parsePEMCert(certData []byte) (*x509.Certificate, error) {
	pemBlock, trailingData := pem.Decode(certData)
	if pemBlock == nil {
		return nil, errors.New("invalid PEM data")
	}
	if len(trailingData) != 0 {
		return nil, errors.New("trailing data after first PEM block")
	}
	return x509.ParseCertificate(pemBlock.Bytes)
}
