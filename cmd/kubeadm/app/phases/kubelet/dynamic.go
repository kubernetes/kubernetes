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
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// EnableDynamicConfigForNode updates the Node's ConfigSource to enable Dynamic Kubelet Configuration, depending on what version the kubelet is
// Used at "kubeadm init", "kubeadm join" and "kubeadm upgrade" time
// This func is ONLY run if the user enables the `DynamicKubeletConfig` feature gate, which is by default off
func EnableDynamicConfigForNode(client clientset.Interface, nodeName string, kubeletVersion *version.Version) error {

	configMapName := kubeadmconstants.GetKubeletConfigMapName(kubeletVersion)
	fmt.Printf("[kubelet] Enabling Dynamic Kubelet Config for Node %q; config sourced from ConfigMap %q in namespace %s\n",
		nodeName, configMapName, metav1.NamespaceSystem)
	fmt.Println("[kubelet] WARNING: The Dynamic Kubelet Config feature is beta, but off by default. It hasn't been well-tested yet at this stage, use with caution.")

	_, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(configMapName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("couldn't get the kubelet configuration ConfigMap: %v", err)
	}

	// Loop on every falsy return. Return with an error if raised. Exit successfully if true is returned.
	return apiclient.PatchNode(client, nodeName, func(n *v1.Node) {
		patchNodeForDynamicConfig(n, configMapName)
	})
}

func patchNodeForDynamicConfig(n *v1.Node, configMapName string) {
	n.Spec.ConfigSource = &v1.NodeConfigSource{
		ConfigMap: &v1.ConfigMapNodeConfigSource{
			Name:             configMapName,
			Namespace:        metav1.NamespaceSystem,
			KubeletConfigKey: kubeadmconstants.KubeletBaseConfigurationConfigMapKey,
		},
	}
}
