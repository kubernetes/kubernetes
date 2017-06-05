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

package file

import (
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
)

// RetrieveValidatedClusterInfo connects to the API Server and makes sure it can talk
// securely to the API Server using the provided CA cert and
// optionally refreshes the cluster-info information from the cluster-info ConfigMap
func RetrieveValidatedClusterInfo(filepath string) (*clientcmdapi.Cluster, error) {
	clusterinfo, err := clientcmd.LoadFromFile(filepath)
	if err != nil {
		return nil, err
	}
	return ValidateClusterInfo(clusterinfo)
}

// ValidateClusterInfo connects to the API Server and makes sure it can talk
// securely to the API Server using the provided CA cert and
// optionally refreshes the cluster-info information from the cluster-info ConfigMap
func ValidateClusterInfo(clusterinfo *clientcmdapi.Config) (*clientcmdapi.Cluster, error) {
	err := validateClusterInfoKubeConfig(clusterinfo)
	if err != nil {
		return nil, err
	}

	// This is the cluster object we've got from the cluster-info KubeConfig file
	defaultCluster := kubeconfigutil.GetClusterFromKubeConfig(clusterinfo)

	// Create a new kubeconfig object from the given, just copy over the server and the CA cert
	// We do this in order to not pick up other possible misconfigurations in the clusterinfo file
	configFromClusterInfo := kubeconfigutil.CreateBasic(
		defaultCluster.Server,
		"kubernetes",
		"", // no user provided
		defaultCluster.CertificateAuthorityData,
	)

	client, err := kubeconfigutil.KubeConfigToClientSet(configFromClusterInfo)
	if err != nil {
		return nil, err
	}

	fmt.Printf("[discovery] Created cluster-info discovery client, requesting info from %q\n", defaultCluster.Server)

	var clusterinfoCM *v1.ConfigMap
	wait.PollInfinite(constants.DiscoveryRetryInterval, func() (bool, error) {
		var err error
		clusterinfoCM, err = client.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsForbidden(err) {
				// If the request is unauthorized, the cluster admin has not granted access to the cluster info configmap for unauthenicated users
				// In that case, trust the cluster admin and do not refresh the cluster-info credentials
				fmt.Printf("[discovery] Could not access the %s ConfigMap for refreshing the cluster-info information, but the TLS cert is valid so proceeding...\n", bootstrapapi.ConfigMapClusterInfo)
				return true, nil
			} else {
				fmt.Printf("[discovery] Failed to validate the API Server's identity, will try again: [%v]\n", err)
				return false, nil
			}
		}
		return true, nil
	})

	// If we couldn't fetch the cluster-info ConfigMap, just return the cluster-info object the user provided
	if clusterinfoCM == nil {
		return defaultCluster, nil
	}

	// We somehow got hold of the ConfigMap, try to read some data from it. If we can't, fallback on the user-provided file
	refreshedBaseKubeConfig, err := tryParseClusterInfoFromConfigMap(clusterinfoCM)
	if err != nil {
		fmt.Printf("[discovery] The %s ConfigMap isn't set up properly (%v), but the TLS cert is valid so proceeding...\n", bootstrapapi.ConfigMapClusterInfo, err)
		return defaultCluster, nil
	}

	fmt.Println("[discovery] Synced cluster-info information from the API Server so we have got the latest information")
	// In an HA world in the future, this will make more sense, because now we've got new information, possibly about new API Servers to talk to
	return kubeconfigutil.GetClusterFromKubeConfig(refreshedBaseKubeConfig), nil
}

// tryParseClusterInfoFromConfigMap tries to parse a kubeconfig file from a ConfigMap key
func tryParseClusterInfoFromConfigMap(cm *v1.ConfigMap) (*clientcmdapi.Config, error) {
	kubeConfigString, ok := cm.Data[bootstrapapi.KubeConfigKey]
	if !ok || len(kubeConfigString) == 0 {
		return nil, fmt.Errorf("no %s key in ConfigMap", bootstrapapi.KubeConfigKey)
	}
	parsedKubeConfig, err := clientcmd.Load([]byte(kubeConfigString))
	if err != nil {
		return nil, fmt.Errorf("couldn't parse the kubeconfig file in the %s ConfigMap: %v", bootstrapapi.ConfigMapClusterInfo, err)
	}
	return parsedKubeConfig, nil
}

// validateClusterInfoKubeConfig makes sure the user-provided cluster-info KubeConfig file is valid
func validateClusterInfoKubeConfig(clusterinfo *clientcmdapi.Config) error {
	if len(clusterinfo.Clusters) < 1 {
		return fmt.Errorf("the provided cluster-info KubeConfig file must have at least one Cluster defined")
	}
	defaultCluster := kubeconfigutil.GetClusterFromKubeConfig(clusterinfo)
	if defaultCluster == nil {
		return fmt.Errorf("the provided cluster-info KubeConfig file must have an unnamed Cluster or a CurrentContext that specifies a non-nil Cluster")
	}
	return nil
}
