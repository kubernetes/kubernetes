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

package kubelet

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
	kubeletconfigscheme "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	kubeletconfigv1beta1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1beta1"
)

// CreateBaseKubeletConfiguration creates base kubelet configuration for dynamic kubelet configuration feature.
func CreateBaseKubeletConfiguration(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {
	fmt.Printf("[kubelet] Uploading a ConfigMap %q in namespace %s with base configuration for the kubelets in the cluster\n",
		kubeadmconstants.KubeletBaseConfigurationConfigMap, metav1.NamespaceSystem)

	_, kubeletCodecs, err := kubeletconfigscheme.NewSchemeAndCodecs()
	if err != nil {
		return err
	}
	kubeletBytes, err := kubeadmutil.MarshalToYamlForCodecs(cfg.KubeletConfiguration.BaseConfig, kubeletconfigv1beta1.SchemeGroupVersion, *kubeletCodecs)
	if err != nil {
		return err
	}

	if err = apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeletBaseConfigurationConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(kubeletBytes),
		},
	}); err != nil {
		return err
	}

	if err := createKubeletBaseConfigMapRBACRules(client); err != nil {
		return fmt.Errorf("error creating base kubelet configmap RBAC rules: %v", err)
	}

	return updateNodeWithConfigMap(client, cfg.NodeName)
}

// ConsumeBaseKubeletConfiguration consumes base kubelet configuration for dynamic kubelet configuration feature.
func ConsumeBaseKubeletConfiguration(nodeName string) error {
	client, err := getLocalNodeTLSBootstrappedClient()
	if err != nil {
		return err
	}

	kubeletCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(kubeadmconstants.KubeletBaseConfigurationConfigMap, metav1.GetOptions{})
	if err != nil {
		return err
	}

	if err := writeInitKubeletConfigToDisk([]byte(kubeletCfg.Data[kubeadmconstants.KubeletBaseConfigurationConfigMapKey])); err != nil {
		return fmt.Errorf("failed to write initial remote configuration of kubelet to disk for node %s: %v", nodeName, err)
	}

	return updateNodeWithConfigMap(client, nodeName)
}

// updateNodeWithConfigMap updates node ConfigSource with KubeletBaseConfigurationConfigMap
func updateNodeWithConfigMap(client clientset.Interface, nodeName string) error {
	fmt.Printf("[kubelet] Using Dynamic Kubelet Config for node %q; config sourced from ConfigMap %q in namespace %s\n",
		nodeName, kubeadmconstants.KubeletBaseConfigurationConfigMap, metav1.NamespaceSystem)

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

		kubeletCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(kubeadmconstants.KubeletBaseConfigurationConfigMap, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		node.Spec.ConfigSource = &v1.NodeConfigSource{
			ConfigMap: &v1.ConfigMapNodeConfigSource{
				Name:             kubeadmconstants.KubeletBaseConfigurationConfigMap,
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

// createKubeletBaseConfigMapRBACRules creates the RBAC rules for exposing the base kubelet ConfigMap in the kube-system namespace to unauthenticated users
func createKubeletBaseConfigMapRBACRules(client clientset.Interface) error {
	if err := apiclient.CreateOrUpdateRole(client, &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeletBaseConfigMapRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		Rules: []rbac.PolicyRule{
			rbachelper.NewRule("get").Groups("").Resources("configmaps").Names(kubeadmconstants.KubeletBaseConfigurationConfigMap).RuleOrDie(),
		},
	}); err != nil {
		return err
	}

	return apiclient.CreateOrUpdateRoleBinding(client, &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeletBaseConfigMapRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     kubeadmconstants.KubeletBaseConfigMapRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: rbac.GroupKind,
				Name: kubeadmconstants.NodesGroup,
			},
			{
				Kind: rbac.GroupKind,
				Name: kubeadmconstants.NodeBootstrapTokenAuthGroup,
			},
		},
	})
}

// getLocalNodeTLSBootstrappedClient waits for the kubelet to perform the TLS bootstrap
// and then creates a client from config file /etc/kubernetes/kubelet.conf
func getLocalNodeTLSBootstrappedClient() (clientset.Interface, error) {
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

// WriteInitKubeletConfigToDiskOnMaster writes base kubelet configuration to disk on master.
func WriteInitKubeletConfigToDiskOnMaster(cfg *kubeadmapi.MasterConfiguration) error {
	fmt.Printf("[kubelet] Writing base configuration of kubelets to disk on master node %s\n", cfg.NodeName)

	_, kubeletCodecs, err := kubeletconfigscheme.NewSchemeAndCodecs()
	if err != nil {
		return err
	}

	kubeletBytes, err := kubeadmutil.MarshalToYamlForCodecs(cfg.KubeletConfiguration.BaseConfig, kubeletconfigv1beta1.SchemeGroupVersion, *kubeletCodecs)
	if err != nil {
		return err
	}

	if err := writeInitKubeletConfigToDisk(kubeletBytes); err != nil {
		return fmt.Errorf("failed to write base configuration of kubelet to disk on master node %s: %v", cfg.NodeName, err)
	}

	return nil
}

func writeInitKubeletConfigToDisk(kubeletConfig []byte) error {
	if err := os.MkdirAll(kubeadmconstants.KubeletBaseConfigurationDir, 0644); err != nil {
		return fmt.Errorf("failed to create directory %q: %v", kubeadmconstants.KubeletBaseConfigurationDir, err)
	}
	baseConfigFile := filepath.Join(kubeadmconstants.KubeletBaseConfigurationDir, kubeadmconstants.KubeletBaseConfigurationFile)
	if err := ioutil.WriteFile(baseConfigFile, kubeletConfig, 0644); err != nil {
		return fmt.Errorf("failed to write initial remote configuration of kubelet into file %q: %v", baseConfigFile, err)
	}
	return nil
}
