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

package kubelet

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/version"
)

// EnableDynamicConfigForNode updates the Node's ConfigSource to enable Dynamic Kubelet Configuration, depending on what version the kubelet is
// Used at "kubeadm init", "kubeadm join" and "kubeadm upgrade" time
// This func is ONLY run if the user enables the `DynamicKubeletConfig` feature gate, which is by default off
func EnableDynamicConfigForNode(client clientset.Interface, nodeName string, kubeletVersion *version.Version) error {

	// If the kubelet version is v1.10.x, exit
	if kubeletVersion.LessThan(kubeadmconstants.MinimumKubeletConfigVersion) {
		return nil
	}

	configMapName := configMapName(kubeletVersion)
	fmt.Printf("[kubelet] Enabling Dynamic Kubelet Config for Node %q; config sourced from ConfigMap %q in namespace %s\n",
		nodeName, configMapName, metav1.NamespaceSystem)
	fmt.Println("[kubelet] WARNING: The Dynamic Kubelet Config feature is alpha and off by default. It hasn't been well-tested yet at this stage, use with caution.")

	// Loop on every falsy return. Return with an error if raised. Exit successfully if true is returned.
	return wait.Poll(kubeadmconstants.APICallRetryInterval, kubeadmconstants.UpdateNodeTimeout, func() (bool, error) {
		node, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		oldData, err := json.Marshal(node)
		if err != nil {
			return false, err
		}

		kubeletCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(configMapName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		node.Spec.ConfigSource = &v1.NodeConfigSource{
			ConfigMap: &v1.ConfigMapNodeConfigSource{
				Name:             configMapName,
				Namespace:        metav1.NamespaceSystem,
				UID:              kubeletCfg.UID,
				KubeletConfigKey: kubeadmconstants.KubeletBaseConfigurationConfigMapKey,
			},
		}

		newData, err := json.Marshal(node)
		if err != nil {
			return false, err
		}

		patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Node{})
		if err != nil {
			return false, err
		}

		if _, err := client.CoreV1().Nodes().Patch(node.Name, types.StrategicMergePatchType, patchBytes); err != nil {
			if apierrs.IsConflict(err) {
				fmt.Println("Temporarily unable to update node metadata due to conflict (will retry)")
				return false, nil
			}
			return false, err
		}

		return true, nil
	})
}

// GetLocalNodeTLSBootstrappedClient waits for the kubelet to perform the TLS bootstrap
// and then creates a client from config file /etc/kubernetes/kubelet.conf
func GetLocalNodeTLSBootstrappedClient() (clientset.Interface, error) {
	fmt.Println("[tlsbootstrap] Waiting for the kubelet to perform the TLS Bootstrap...")

	kubeletKubeConfig := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.KubeletKubeConfigFileName)

	// Loop on every falsy return. Return with an error if raised. Exit successfully if true is returned.
	err := wait.PollImmediateInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		_, err := os.Stat(kubeletKubeConfig)
		return (err == nil), nil
	})
	if err != nil {
		return nil, err
	}

	return kubeconfigutil.ClientSetFromFile(kubeletKubeConfig)
}
