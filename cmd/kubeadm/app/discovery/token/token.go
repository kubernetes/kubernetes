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
	"context"
	"fmt"
	"time"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	tokenjws "k8s.io/cluster-bootstrap/token/jws"
	"k8s.io/klog/v2"

	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pubkeypin"
)

// BootstrapUser defines bootstrap user name
const BootstrapUser = "token-bootstrap-client"

// RetrieveValidatedConfigInfo connects to the API Server and tries to fetch the cluster-info ConfigMap
// It then makes sure it can trust the API Server by looking at the JWS-signed tokens and (if CACertHashes is not empty)
// validating the cluster CA against a set of pinned public keys
func RetrieveValidatedConfigInfo(dryRunClient clientset.Interface, cfg *kubeadmapi.Discovery, timeout time.Duration) (*clientcmdapi.Config, error) {
	isDryRun := dryRunClient != nil
	isTesting := false
	return retrieveValidatedConfigInfo(dryRunClient, cfg, constants.DiscoveryRetryInterval, timeout, isDryRun, isTesting)
}

// retrieveValidatedConfigInfo is a private implementation of RetrieveValidatedConfigInfo.
// It accepts an optional clientset that can be used for testing purposes.
func retrieveValidatedConfigInfo(client clientset.Interface, cfg *kubeadmapi.Discovery, interval, timeout time.Duration, isDryRun, isTesting bool) (*clientcmdapi.Config, error) {
	var err error

	// Make sure the interval is not bigger than the duration
	if interval > timeout {
		interval = timeout
	}

	endpoint := cfg.BootstrapToken.APIServerEndpoint
	insecureBootstrapConfig := BuildInsecureBootstrapKubeConfig(endpoint)
	clusterName := insecureBootstrapConfig.Contexts[insecureBootstrapConfig.CurrentContext].Cluster

	klog.V(1).Infof("[discovery] Created cluster-info discovery client, requesting info from %q", endpoint)
	if !isDryRun && !isTesting {
		client, err = kubeconfigutil.ToClientSet(insecureBootstrapConfig)
		if err != nil {
			return nil, err
		}
	}
	insecureClusterInfo, err := getClusterInfo(client, cfg, interval, timeout, isDryRun)
	if err != nil {
		return nil, err
	}

	// Load the CACertHashes into a pubkeypin.Set
	pubKeyPins := pubkeypin.NewSet()
	if err := pubKeyPins.Allow(cfg.BootstrapToken.CACertHashes...); err != nil {
		return nil, errors.Wrap(err, "invalid discovery token CA certificate hash")
	}

	token, err := bootstraptokenv1.NewBootstrapTokenString(cfg.BootstrapToken.Token)
	if err != nil {
		return nil, err
	}

	// Validate the token in the cluster info
	insecureKubeconfigBytes, err := validateClusterInfoToken(insecureClusterInfo, token)
	if err != nil {
		return nil, err
	}

	// Load the insecure config
	insecureConfig, err := clientcmd.Load(insecureKubeconfigBytes)
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't parse the kubeconfig file in the %s ConfigMap", bootstrapapi.ConfigMapClusterInfo)
	}

	// The ConfigMap should contain a single cluster
	if len(insecureConfig.Clusters) != 1 {
		return nil, errors.Errorf("expected the kubeconfig file in the %s ConfigMap to have a single cluster, but it had %d", bootstrapapi.ConfigMapClusterInfo, len(insecureConfig.Clusters))
	}

	// If no TLS root CA pinning was specified, we're done
	if pubKeyPins.Empty() {
		klog.V(1).Infof("[discovery] Cluster info signature and contents are valid and no TLS pinning was specified, will use API Server %q", endpoint)
		return insecureConfig, nil
	}

	// Load and validate the cluster CA from the insecure kubeconfig
	clusterCABytes, err := validateClusterCA(insecureConfig, pubKeyPins)
	if err != nil {
		return nil, err
	}

	// Now that we know the cluster CA, connect back a second time validating with that CA
	secureBootstrapConfig := buildSecureBootstrapKubeConfig(endpoint, clusterCABytes, clusterName)

	klog.V(1).Infof("[discovery] Requesting info from %q again to validate TLS against the pinned public key", endpoint)
	if !isDryRun && !isTesting {
		client, err = kubeconfigutil.ToClientSet(secureBootstrapConfig)
		if err != nil {
			return nil, err
		}
	}
	secureClusterInfo, err := getClusterInfo(client, cfg, interval, timeout, isDryRun)
	if err != nil {
		return nil, err
	}

	// Pull the kubeconfig from the securely-obtained ConfigMap and validate that it's the same as what we found the first time
	secureKubeconfigBytes := []byte(secureClusterInfo.Data[bootstrapapi.KubeConfigKey])
	if !bytes.Equal(secureKubeconfigBytes, insecureKubeconfigBytes) {
		return nil, errors.Errorf("the second kubeconfig from the %s ConfigMap (using validated TLS) was different from the first", bootstrapapi.ConfigMapClusterInfo)
	}

	secureKubeconfig, err := clientcmd.Load(secureKubeconfigBytes)
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't parse the kubeconfig file in the %s ConfigMap", bootstrapapi.ConfigMapClusterInfo)
	}

	klog.V(1).Infof("[discovery] Cluster info signature and contents are valid and TLS certificate validates against pinned roots, will use API Server %q", endpoint)

	return secureKubeconfig, nil
}

// BuildInsecureBootstrapKubeConfig makes a kubeconfig object that connects insecurely to the API Server for bootstrapping purposes
func BuildInsecureBootstrapKubeConfig(endpoint string) *clientcmdapi.Config {
	controlPlaneEndpoint := fmt.Sprintf("https://%s", endpoint)
	clusterName := kubeadmapiv1.DefaultClusterName
	bootstrapConfig := kubeconfigutil.CreateBasic(controlPlaneEndpoint, clusterName, BootstrapUser, []byte{})
	bootstrapConfig.Clusters[clusterName].InsecureSkipTLSVerify = true
	return bootstrapConfig
}

// buildSecureBootstrapKubeConfig makes a kubeconfig object that connects securely to the API Server for bootstrapping purposes (validating with the specified CA)
func buildSecureBootstrapKubeConfig(endpoint string, caCert []byte, clustername string) *clientcmdapi.Config {
	controlPlaneEndpoint := fmt.Sprintf("https://%s", endpoint)
	bootstrapConfig := kubeconfigutil.CreateBasic(controlPlaneEndpoint, clustername, BootstrapUser, caCert)
	return bootstrapConfig
}

// validateClusterInfoToken validates that the JWS token present in the cluster info ConfigMap is valid
func validateClusterInfoToken(insecureClusterInfo *v1.ConfigMap, token *bootstraptokenv1.BootstrapTokenString) ([]byte, error) {
	insecureKubeconfigString, ok := insecureClusterInfo.Data[bootstrapapi.KubeConfigKey]
	if !ok || len(insecureKubeconfigString) == 0 {
		return nil, errors.Errorf("there is no %s key in the %s ConfigMap. This API Server isn't set up for token bootstrapping, can't connect",
			bootstrapapi.KubeConfigKey, bootstrapapi.ConfigMapClusterInfo)
	}

	detachedJWSToken, ok := insecureClusterInfo.Data[bootstrapapi.JWSSignatureKeyPrefix+token.ID]
	if !ok || len(detachedJWSToken) == 0 {
		return nil, errors.Errorf("token id %q is invalid for this cluster or it has expired. Use \"kubeadm token create\" on the control-plane node to create a new valid token", token.ID)
	}

	if !tokenjws.DetachedTokenIsValid(detachedJWSToken, insecureKubeconfigString, token.ID, token.Secret) {
		return nil, errors.New("failed to verify JWS signature of received cluster info object, can't trust this API Server")
	}

	return []byte(insecureKubeconfigString), nil
}

// validateClusterCA validates the cluster CA found in the insecure kubeconfig
func validateClusterCA(insecureConfig *clientcmdapi.Config, pubKeyPins *pubkeypin.Set) ([]byte, error) {
	var clusterCABytes []byte
	for _, cluster := range insecureConfig.Clusters {
		clusterCABytes = cluster.CertificateAuthorityData
	}

	clusterCAs, err := certutil.ParseCertsPEM(clusterCABytes)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse cluster CA from the %s ConfigMap", bootstrapapi.ConfigMapClusterInfo)
	}

	// Validate the cluster CA public key against the pinned set
	err = pubKeyPins.CheckAny(clusterCAs)
	if err != nil {
		return nil, errors.Wrapf(err, "cluster CA found in %s ConfigMap is invalid", bootstrapapi.ConfigMapClusterInfo)
	}

	return clusterCABytes, nil
}

// getClusterInfo requests the cluster-info ConfigMap with the provided client.
func getClusterInfo(client clientset.Interface, cfg *kubeadmapi.Discovery, interval, duration time.Duration, dryRun bool) (*v1.ConfigMap, error) {
	var (
		cm        *v1.ConfigMap
		err       error
		lastError error
	)

	err = wait.PollUntilContextTimeout(context.Background(),
		interval, duration, true,
		func(ctx context.Context) (bool, error) {
			token, err := bootstraptokenv1.NewBootstrapTokenString(cfg.BootstrapToken.Token)
			if err != nil {
				lastError = errors.Wrapf(err, "could not construct token string for token: %s",
					cfg.BootstrapToken.Token)
				return true, lastError
			}

			klog.V(1).Infof("[discovery] Waiting for the cluster-info ConfigMap to receive a JWS signature"+
				" for token ID %q", token.ID)

			cm, err = client.CoreV1().ConfigMaps(metav1.NamespacePublic).
				Get(context.Background(), bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
			if err != nil {
				lastError = errors.Wrapf(err, "failed to request the cluster-info ConfigMap")
				klog.V(1).Infof("[discovery] Retrying due to error: %v", lastError)
				return false, nil
			}
			// Even if the ConfigMap is available the JWS signature is patched-in a bit later.
			if _, ok := cm.Data[bootstrapapi.JWSSignatureKeyPrefix+token.ID]; !ok {
				lastError = errors.Errorf("could not find a JWS signature in the cluster-info ConfigMap"+
					" for token ID %q", token.ID)
				if dryRun {
					// Assume the user is dry-running with a token that will never appear in the cluster-info
					// ConfigMap. Use the default dry-run token and CA cert hash.
					mutateTokenDiscoveryForDryRun(cfg)
					return false, nil
				}
				klog.V(1).Infof("[discovery] Retrying due to error: %v", lastError)
				return false, nil
			}
			return true, nil
		})
	if err != nil {
		return nil, lastError
	}

	return cm, nil
}

// mutateTokenDiscoveryForDryRun mutates the JoinConfiguration.Discovery so that it includes a dry-run token
// CA cert hash and fake API server endpoint to comply with the fake "cluster-info" ConfigMap
// that this reactor returns. The information here should be in sync with what the GetClusterInfoReactor()
// dry-run reactor does.
func mutateTokenDiscoveryForDryRun(cfg *kubeadmapi.Discovery) {
	const (
		tokenID     = "abcdef"
		tokenSecret = "abcdef0123456789"
		caHash      = "sha256:3b793efefe27a19f93b0fbe6e637e9c41d0dde8a377d6ab1c0f656bf1136dd8a"
		endpoint    = "https://192.168.0.101:6443"
	)

	token := fmt.Sprintf("%s.%s", tokenID, tokenSecret)
	klog.Warningf("[dryrun] Mutating the JoinConfiguration.Discovery.BootstrapToken to satisfy "+
		"the dry-run without a real cluster-info ConfigMap:\n"+
		"  Token: %s\n  CACertHash: %s\n  APIServerEndpoint: %s\n",
		token, caHash, endpoint)
	if cfg.BootstrapToken == nil {
		cfg.BootstrapToken = &kubeadmapi.BootstrapTokenDiscovery{}
	}
	cfg.BootstrapToken.Token = token
	cfg.BootstrapToken.CACertHashes = append(cfg.BootstrapToken.CACertHashes, caHash)
	cfg.BootstrapToken.APIServerEndpoint = endpoint
}
