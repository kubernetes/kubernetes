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
	"io/ioutil"

	"github.com/pkg/errors"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// RetrieveValidatedConfigInfo connects to the API Server and makes sure it can talk
// securely to the API Server using the provided CA cert and
// optionally refreshes the cluster-info information from the cluster-info ConfigMap
func RetrieveValidatedConfigInfo(filepath, clustername string) (*clientcmdapi.Config, error) {
	config, err := clientcmd.LoadFromFile(filepath)
	if err != nil {
		return nil, err
	}
	return ValidateConfigInfo(config, clustername)
}

// ValidateConfigInfo connects to the API Server and makes sure it can talk
// securely to the API Server using the provided CA cert/client certificates  and
// optionally refreshes the cluster-info information from the cluster-info ConfigMap
func ValidateConfigInfo(config *clientcmdapi.Config, clustername string) (*clientcmdapi.Config, error) {
	err := validateKubeConfig(config)
	if err != nil {
		return nil, err
	}

	// This is the cluster object we've got from the cluster-info kubeconfig file
	defaultCluster := kubeconfigutil.GetClusterFromKubeConfig(config)

	// Create a new kubeconfig object from the given, just copy over the server and the CA cert
	// We do this in order to not pick up other possible misconfigurations in the clusterinfo file
	kubeconfig := kubeconfigutil.CreateBasic(
		defaultCluster.Server,
		clustername,
		"", // no user provided
		defaultCluster.CertificateAuthorityData,
	)
	// load pre-existing client certificates
	if config.Contexts[config.CurrentContext] != nil && len(config.AuthInfos) > 0 {
		user := config.Contexts[config.CurrentContext].AuthInfo
		authInfo, ok := config.AuthInfos[user]
		if !ok || authInfo == nil {
			return nil, errors.Errorf("empty settings for user %q", user)
		}
		if len(authInfo.ClientCertificateData) == 0 && len(authInfo.ClientCertificate) != 0 {
			clientCert, err := ioutil.ReadFile(authInfo.ClientCertificate)
			if err != nil {
				return nil, err
			}
			authInfo.ClientCertificateData = clientCert
		}
		if len(authInfo.ClientKeyData) == 0 && len(authInfo.ClientKey) != 0 {
			clientKey, err := ioutil.ReadFile(authInfo.ClientKey)
			if err != nil {
				return nil, err
			}
			authInfo.ClientKeyData = clientKey
		}

		if len(authInfo.ClientCertificateData) == 0 || len(authInfo.ClientKeyData) == 0 {
			return nil, errors.New("couldn't read authentication info from the given kubeconfig file")
		}
		kubeconfig = kubeconfigutil.CreateWithCerts(
			defaultCluster.Server,
			clustername,
			"", // no user provided
			defaultCluster.CertificateAuthorityData,
			authInfo.ClientKeyData,
			authInfo.ClientCertificateData,
		)
	}

	client, err := kubeconfigutil.ToClientSet(kubeconfig)
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
				// If the request is unauthorized, the cluster admin has not granted access to the cluster info configmap for unauthenticated users
				// In that case, trust the cluster admin and do not refresh the cluster-info credentials
				fmt.Printf("[discovery] Could not access the %s ConfigMap for refreshing the cluster-info information, but the TLS cert is valid so proceeding...\n", bootstrapapi.ConfigMapClusterInfo)
				return true, nil
			}
			fmt.Printf("[discovery] Failed to validate the API Server's identity, will try again: [%v]\n", err)
			return false, nil
		}
		return true, nil
	})

	// If we couldn't fetch the cluster-info ConfigMap, just return the cluster-info object the user provided
	if clusterinfoCM == nil {
		return kubeconfig, nil
	}

	// We somehow got hold of the ConfigMap, try to read some data from it. If we can't, fallback on the user-provided file
	refreshedBaseKubeConfig, err := tryParseClusterInfoFromConfigMap(clusterinfoCM)
	if err != nil {
		fmt.Printf("[discovery] The %s ConfigMap isn't set up properly (%v), but the TLS cert is valid so proceeding...\n", bootstrapapi.ConfigMapClusterInfo, err)
		return kubeconfig, nil
	}

	fmt.Println("[discovery] Synced cluster-info information from the API Server so we have got the latest information")
	// In an HA world in the future, this will make more sense, because now we've got new information, possibly about new API Servers to talk to
	return refreshedBaseKubeConfig, nil
}

// tryParseClusterInfoFromConfigMap tries to parse a kubeconfig file from a ConfigMap key
func tryParseClusterInfoFromConfigMap(cm *v1.ConfigMap) (*clientcmdapi.Config, error) {
	kubeConfigString, ok := cm.Data[bootstrapapi.KubeConfigKey]
	if !ok || len(kubeConfigString) == 0 {
		return nil, errors.Errorf("no %s key in ConfigMap", bootstrapapi.KubeConfigKey)
	}
	parsedKubeConfig, err := clientcmd.Load([]byte(kubeConfigString))
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't parse the kubeconfig file in the %s ConfigMap", bootstrapapi.ConfigMapClusterInfo)
	}
	return parsedKubeConfig, nil
}

// validateKubeConfig makes sure the user-provided kubeconfig file is valid
func validateKubeConfig(config *clientcmdapi.Config) error {
	if len(config.Clusters) < 1 {
		return errors.New("the provided cluster-info kubeconfig file must have at least one Cluster defined")
	}
	defaultCluster := kubeconfigutil.GetClusterFromKubeConfig(config)
	if defaultCluster == nil {
		return errors.New("the provided cluster-info kubeconfig file must have an unnamed Cluster or a CurrentContext that specifies a non-nil Cluster")
	}
	return clientcmd.Validate(*config)
}
