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
	"fmt"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
	"k8s.io/kubernetes/pkg/controller/bootstrap"
)

const BootstrapUser = "token-bootstrap-client"

// RetrieveValidatedClusterInfo connects to the API Server and tries to fetch the cluster-info ConfigMap
// It then makes sure it can trust the API Server by looking at the JWS-signed tokens
func RetrieveValidatedClusterInfo(discoveryToken string, tokenAPIServers []string) (*clientcmdapi.Cluster, error) {

	tokenId, tokenSecret, err := tokenutil.ParseToken(discoveryToken)
	if err != nil {
		return nil, err
	}

	// The function below runs for every endpoint, and all endpoints races with each other.
	// The endpoint that wins the race and completes the task first gets its kubeconfig returned below
	baseKubeConfig := runForEndpointsAndReturnFirst(tokenAPIServers, func(endpoint string) (*clientcmdapi.Config, error) {

		bootstrapConfig := buildInsecureBootstrapKubeConfig(endpoint)
		clusterName := bootstrapConfig.Contexts[bootstrapConfig.CurrentContext].Cluster

		client, err := kubeconfigutil.KubeConfigToClientSet(bootstrapConfig)
		if err != nil {
			return nil, err
		}

		fmt.Printf("[discovery] Created cluster-info discovery client, requesting info from %q\n", bootstrapConfig.Clusters[clusterName].Server)

		var clusterinfo *v1.ConfigMap
		wait.PollInfinite(constants.DiscoveryRetryInterval, func() (bool, error) {
			var err error
			clusterinfo, err = client.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
			if err != nil {
				fmt.Printf("[discovery] Failed to request cluster info, will try again: [%s]\n", err)
				return false, nil
			}
			return true, nil
		})

		kubeConfigString, ok := clusterinfo.Data[bootstrapapi.KubeConfigKey]
		if !ok || len(kubeConfigString) == 0 {
			return nil, fmt.Errorf("there is no %s key in the %s ConfigMap. This API Server isn't set up for token bootstrapping, can't connect", bootstrapapi.KubeConfigKey, bootstrapapi.ConfigMapClusterInfo)
		}
		detachedJWSToken, ok := clusterinfo.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenId]
		if !ok || len(detachedJWSToken) == 0 {
			return nil, fmt.Errorf("there is no JWS signed token in the %s ConfigMap. This token id %q is invalid for this cluster, can't connect", bootstrapapi.ConfigMapClusterInfo, tokenId)
		}
		if !bootstrap.DetachedTokenIsValid(detachedJWSToken, kubeConfigString, tokenId, tokenSecret) {
			return nil, fmt.Errorf("failed to verify JWS signature of received cluster info object, can't trust this API Server")
		}

		finalConfig, err := clientcmd.Load([]byte(kubeConfigString))
		if err != nil {
			return nil, fmt.Errorf("couldn't parse the kubeconfig file in the %s configmap: %v", bootstrapapi.ConfigMapClusterInfo, err)
		}

		fmt.Printf("[discovery] Cluster info signature and contents are valid, will use API Server %q\n", bootstrapConfig.Clusters[clusterName].Server)
		return finalConfig, nil
	})

	return kubeconfigutil.GetClusterFromKubeConfig(baseKubeConfig), nil
}

// buildInsecureBootstrapKubeConfig makes a KubeConfig object that connects insecurely to the API Server for bootstrapping purposes
func buildInsecureBootstrapKubeConfig(endpoint string) *clientcmdapi.Config {
	masterEndpoint := fmt.Sprintf("https://%s", endpoint)
	clusterName := "kubernetes"
	bootstrapConfig := kubeconfigutil.CreateBasic(masterEndpoint, clusterName, BootstrapUser, []byte{})
	bootstrapConfig.Clusters[clusterName].InsecureSkipTLSVerify = true
	return bootstrapConfig
}

// runForEndpointsAndReturnFirst loops the endpoints slice and let's the endpoints race for connecting to the master
func runForEndpointsAndReturnFirst(endpoints []string, fetchKubeConfigFunc func(string) (*clientcmdapi.Config, error)) *clientcmdapi.Config {
	stopChan := make(chan struct{})
	var resultingKubeConfig *clientcmdapi.Config
	var once sync.Once
	var wg sync.WaitGroup
	for _, endpoint := range endpoints {
		wg.Add(1)
		go func(apiEndpoint string) {
			defer wg.Done()
			wait.Until(func() {
				fmt.Printf("[discovery] Trying to connect to API Server %q\n", apiEndpoint)
				cfg, err := fetchKubeConfigFunc(apiEndpoint)
				if err != nil {
					fmt.Printf("[discovery] Failed to connect to API Server %q: %v\n", apiEndpoint, err)
					return
				}
				fmt.Printf("[discovery] Successfully established connection with API Server %q\n", apiEndpoint)

				// connection established, stop all wait threads
				once.Do(func() {
					close(stopChan)
					resultingKubeConfig = cfg
				})
			}, constants.DiscoveryRetryInterval, stopChan)
		}(endpoint)
	}
	wg.Wait()
	return resultingKubeConfig
}
