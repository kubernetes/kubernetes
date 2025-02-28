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
	"context"
	"time"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// RetrieveValidatedConfigInfo connects to the API Server and makes sure it can talk
// securely to the API Server using the provided CA cert and
// optionally refreshes the cluster-info information from the cluster-info ConfigMap
func RetrieveValidatedConfigInfo(filepath string, discoveryTimeout time.Duration) (*clientcmdapi.Config, error) {
	config, err := clientcmd.LoadFromFile(filepath)
	if err != nil {
		return nil, err
	}
	return ValidateConfigInfo(config, discoveryTimeout)
}

// ValidateConfigInfo connects to the API Server and makes sure it can talk
// securely to the API Server using the provided CA cert/client certificates  and
// optionally refreshes the cluster-info information from the cluster-info ConfigMap
func ValidateConfigInfo(config *clientcmdapi.Config, discoveryTimeout time.Duration) (*clientcmdapi.Config, error) {
	if len(config.Clusters) < 1 {
		return nil, errors.New("the provided kubeconfig file must have at least one Cluster defined")
	}
	currentClusterName, currentCluster := kubeconfigutil.GetClusterFromKubeConfig(config)
	if currentCluster == nil {
		return nil, errors.New("the provided kubeconfig file must have a unnamed Cluster or a CurrentContext that specifies a non-nil Cluster")
	}
	if err := clientcmd.Validate(*config); err != nil {
		return nil, err
	}

	// If the kubeconfig points to a file for the CA, make sure the CA file contents are embedded
	if err := kubeconfigutil.EnsureCertificateAuthorityIsEmbedded(currentCluster); err != nil {
		return nil, err
	}

	// If the discovery file config contains authentication credentials
	if kubeconfigutil.HasAuthenticationCredentials(config) {
		klog.V(1).Info("[discovery] Using authentication credentials from the discovery file for validating TLS connection")

		// We should ensure that all the authentication info is embedded in config file, so everything will work also when
		// the kubeconfig file will be stored in /etc/kubernetes/boostrap-kubelet.conf
		if err := kubeconfigutil.EnsureAuthenticationInfoAreEmbedded(config); err != nil {
			return nil, err
		}
	} else {
		// If the discovery file config does not contains authentication credentials
		klog.V(1).Info("[discovery] Discovery file does not contains authentication credentials, using unauthenticated request for validating TLS connection")
	}

	// Try to read the cluster-info config map; this step was required by the original design in order
	// to validate the TLS connection to the server early in the process
	client, err := kubeconfigutil.ToClientSet(config)
	if err != nil {
		return nil, err
	}

	klog.V(1).Infof("[discovery] Created cluster-info discovery client, requesting info from %q\n", currentCluster.Server)

	var clusterinfoCM *v1.ConfigMap

	var lastError error
	err = wait.PollUntilContextTimeout(context.Background(),
		constants.DiscoveryRetryInterval, discoveryTimeout,
		true, func(_ context.Context) (bool, error) {
			clusterinfoCM, lastError = client.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(context.TODO(), bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
			if lastError != nil {
				if apierrors.IsForbidden(lastError) {
					// If the request fails with a forbidden error, the cluster admin has not granted access to the cluster info configmap for anonymous clients.
					// In that case, trust the cluster admin and do not refresh the cluster-info data
					klog.Warningf("[discovery] Could not access the %s ConfigMap for refreshing the cluster-info information, but the TLS cert is valid so proceeding...\n", bootstrapapi.ConfigMapClusterInfo)
					return true, nil
				}
				klog.V(1).Infof("[discovery] Error reading the %s ConfigMap, will try again: %v\n", bootstrapapi.ConfigMapClusterInfo, lastError)
				return false, nil
			}
			return true, nil
		})
	if err != nil {
		return nil, errors.Wrapf(lastError, "Abort reading the %s ConfigMap after timeout of %v",
			bootstrapapi.ConfigMapClusterInfo, discoveryTimeout)
	}

	// If we couldn't fetch the cluster-info ConfigMap, just return the cluster-info object the user provided
	if clusterinfoCM == nil {
		return config, nil
	}

	// We somehow got hold of the ConfigMap, try to read some data from it. If we can't, fallback on the user-provided file
	refreshedBaseKubeConfig, err := tryParseClusterInfoFromConfigMap(clusterinfoCM)
	if err != nil {
		klog.V(1).Infof("[discovery] The %s ConfigMap isn't set up properly (%v), but the TLS cert is valid so proceeding...\n", bootstrapapi.ConfigMapClusterInfo, err)
		return config, nil
	}

	_, refreshedCluster := kubeconfigutil.GetClusterFromKubeConfig(refreshedBaseKubeConfig)
	if currentCluster.Server != refreshedCluster.Server {
		klog.Warningf("[discovery] the API Server endpoint %q in use is different from the endpoint %q which defined in the %s ConfigMap", currentCluster.Server, refreshedCluster.Server, bootstrapapi.ConfigMapClusterInfo)
	}

	if len(currentCluster.CertificateAuthorityData) == 0 && len(refreshedCluster.CertificateAuthorityData) > 0 {
		config.Clusters[currentClusterName].CertificateAuthorityData = refreshedCluster.CertificateAuthorityData
		klog.V(1).Infof("[discovery] Synced CertificateAuthorityData from the %s ConfigMap", bootstrapapi.ConfigMapClusterInfo)
	}

	return config, nil
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
