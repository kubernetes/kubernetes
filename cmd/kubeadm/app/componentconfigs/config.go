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

package componentconfigs

import (
	"github.com/pkg/errors"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
)

// GetFromKubeletConfigMap returns the pointer to the ComponentConfig API object read from the kubelet-config-version
// ConfigMap map stored in the cluster
func GetFromKubeletConfigMap(client clientset.Interface, version *version.Version) (runtime.Object, error) {

	// Read the ConfigMap from the cluster based on what version the kubelet is
	configMapName := kubeadmconstants.GetKubeletConfigMapName(version)
	kubeletCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(configMapName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	kubeletConfigData, ok := kubeletCfg.Data[kubeadmconstants.KubeletBaseConfigurationConfigMapKey]
	if !ok {
		return nil, errors.Errorf("unexpected error when reading %s ConfigMap: %s key value pair missing",
			configMapName, kubeadmconstants.KubeletBaseConfigurationConfigMapKey)
	}

	// Decodes the kubeletConfigData into the internal component config
	obj := &kubeletconfig.KubeletConfiguration{}
	err = unmarshalObject(obj, []byte(kubeletConfigData))
	if err != nil {
		return nil, err
	}

	return obj, nil
}

// GetFromKubeProxyConfigMap returns the pointer to the ComponentConfig API object read from the kube-proxy
// ConfigMap map stored in the cluster
func GetFromKubeProxyConfigMap(client clientset.Interface, version *version.Version) (runtime.Object, error) {

	// Read the ConfigMap from the cluster
	kubeproxyCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(kubeadmconstants.KubeProxyConfigMap, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	kubeproxyConfigData, ok := kubeproxyCfg.Data[kubeadmconstants.KubeProxyConfigMapKey]
	if !ok {
		return nil, errors.Errorf("unexpected error when reading %s ConfigMap: %s key value pair missing",
			kubeadmconstants.KubeProxyConfigMap, kubeadmconstants.KubeProxyConfigMapKey)
	}

	// Decodes the Config map dat into the internal component config
	obj := &kubeproxyconfig.KubeProxyConfiguration{}
	err = unmarshalObject(obj, []byte(kubeproxyConfigData))
	if err != nil {
		return nil, err
	}

	return obj, nil
}
