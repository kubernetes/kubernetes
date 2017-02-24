/*
Copyright 2017 The Kubernetes Authors.

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
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	//tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
	"k8s.io/kubernetes/pkg/controller/bootstrap"
)

// the amount of time to wait between each request to the discovery API
const discoveryRetryTimeout = 5 * time.Second

const (
	ClusterName   = "kubernetes"
	TokenUser     = "kubelet-token"
	BootstrapUser = "token-bootstrap-client"
)

// RetrieveValidatedTokenKubeConfig TODO
func RetrieveValidatedTokenKubeConfig(d *kubeadmapi.TokenDiscovery) *clientcmdapi.Config {

	// The function below runs for every endpoint, and all endpoints races with each other.
	// The endpoint that wins the race and completes the task first gets its kubeconfig returned below
	baseKubeConfig := runForEndpointsAndReturnFirst(d.Addresses, func(endpoint string) (*clientcmdapi.Config, error) {

		bootstrapConfig := buildInsecureBootstrapKubeConfig(endpoint)
		client, err := kubeconfigutil.KubeConfigToClientSet(bootstrapConfig)
		if err != nil {
			return nil, err
		}

		fmt.Printf("[discovery] Created cluster-info discovery client, requesting info from %q\n", bootstrapConfig.Clusters[ClusterName].Server)

		var clusterinfo *v1.ConfigMap
		wait.PollInfinite(discoveryRetryTimeout, func() (bool, error) {
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
		detachedJWSToken, ok := clusterinfo.Data[bootstrapapi.JWSSignatureKeyPrefix+d.ID]
		if !ok || len(detachedJWSToken) == 0 {
			return nil, fmt.Errorf("there is no JWS signed token in the %s ConfigMap. This token id %q is invalid for this cluster, can't connect", bootstrapapi.ConfigMapClusterInfo, d.ID)
		}
		if !bootstrap.DetachedTokenIsValid(detachedJWSToken, kubeConfigString, d.ID, d.Secret) {
			return nil, fmt.Errorf("failed to verify JWS signature of received cluster info object, can't trust this API Server")
		}

		finalConfig, err := clientcmd.Load([]byte(kubeConfigString))
		if err != nil {
			return nil, fmt.Errorf("couldn't parse the kubeconfig file in the %s configmap: %v", bootstrapapi.ConfigMapClusterInfo, err)
		}

		// TODO: Print summary info about the CA certificate, along with the the checksum signature
		// we also need an ability for the user to configure the client to validate received CA cert against a checksum
		fmt.Printf("[discovery] Cluster info signature and contents are valid, will use API Server %q\n", bootstrapConfig.Clusters[ClusterName].Server)
		return finalConfig, nil
	})

	clusterName := baseKubeConfig.Contexts[baseKubeConfig.CurrentContext].Cluster
	return kubeconfigutil.CreateWithToken(
		baseKubeConfig.Clusters[clusterName].Server,
		ClusterName,
		TokenUser,
		baseKubeConfig.Clusters[clusterName].CertificateAuthorityData,
		d.ID+"."+d.Secret, // TODO: fixup and use BearerToken
	)
}

// buildInsecureBootstrapKubeConfig makes a KubeConfig object that connects insecurely to the API Server for bootstrapping purposes
func buildInsecureBootstrapKubeConfig(endpoint string) *clientcmdapi.Config {
	masterEndpoint := fmt.Sprintf("https://%s", endpoint)
	bootstrapConfig := kubeconfigutil.CreateBasic(masterEndpoint, ClusterName, BootstrapUser, []byte{})
	bootstrapConfig.Clusters[ClusterName].InsecureSkipTLSVerify = true
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
			}, discoveryRetryTimeout, stopChan)
		}(endpoint)
	}
	wg.Wait()
	return resultingKubeConfig
}
