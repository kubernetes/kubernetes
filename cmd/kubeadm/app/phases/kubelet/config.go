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
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// WriteConfigToDisk writes the kubelet config object down to a file
// Used at "kubeadm init" and "kubeadm upgrade" time
func WriteConfigToDisk(cfg *kubeadmapi.ClusterConfiguration, kubeletDir string) error {
	kubeletCfg, ok := cfg.ComponentConfigs[componentconfigs.KubeletGroup]
	if !ok {
		return errors.New("no kubelet component config found")
	}

	if err := kubeletCfg.Mutate(); err != nil {
		return err
	}

	kubeletBytes, err := kubeletCfg.Marshal()
	if err != nil {
		return err
	}

	return writeConfigBytesToDisk(kubeletBytes, kubeletDir)
}

// CreateConfigMap creates a ConfigMap with the generic kubelet configuration.
// Used at "kubeadm init" and "kubeadm upgrade" time
func CreateConfigMap(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface) error {
	k8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return err
	}

	// TODO: cleanup after UnversionedKubeletConfigMap goes GA:
	// https://github.com/kubernetes/kubeadm/issues/1582
	legacyKubeletCM := !features.Enabled(cfg.FeatureGates, features.UnversionedKubeletConfigMap)
	configMapName := kubeadmconstants.GetKubeletConfigMapName(k8sVersion, legacyKubeletCM)
	fmt.Printf("[kubelet] Creating a ConfigMap %q in namespace %s with the configuration for the kubelets in the cluster\n", configMapName, metav1.NamespaceSystem)
	if legacyKubeletCM {
		fmt.Printf("NOTE: The %q naming of the kubelet ConfigMap is deprecated. "+
			"Once the UnversionedKubeletConfigMap feature gate graduates to Beta the default name will become just %q. "+
			"Kubeadm upgrade will handle this transition transparently.\n",
			configMapName, kubeadmconstants.KubeletBaseConfigurationConfigMap)
	}

	kubeletCfg, ok := cfg.ComponentConfigs[componentconfigs.KubeletGroup]
	if !ok {
		return errors.New("no kubelet component config found in the active component config set")
	}

	kubeletBytes, err := kubeletCfg.Marshal()
	if err != nil {
		return err
	}

	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(kubeletBytes),
		},
	}

	if !kubeletCfg.IsUserSupplied() {
		componentconfigs.SignConfigMap(configMap)
	}

	if err := apiclient.CreateOrUpdateConfigMap(client, configMap); err != nil {
		return err
	}

	if err := createConfigMapRBACRules(client, k8sVersion, legacyKubeletCM); err != nil {
		return errors.Wrap(err, "error creating kubelet configuration configmap RBAC rules")
	}
	return nil
}

// createConfigMapRBACRules creates the RBAC rules for exposing the base kubelet ConfigMap in the kube-system namespace to unauthenticated users
// TODO: Remove the legacy arg once UnversionedKubeletConfigMap graduates to GA:
// https://github.com/kubernetes/kubeadm/issues/1582
func createConfigMapRBACRules(client clientset.Interface, k8sVersion *version.Version, legacy bool) error {
	if err := apiclient.CreateOrUpdateRole(client, &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapRBACName(k8sVersion, legacy),
			Namespace: metav1.NamespaceSystem,
		},
		Rules: []rbac.PolicyRule{
			{
				Verbs:         []string{"get"},
				APIGroups:     []string{""},
				Resources:     []string{"configmaps"},
				ResourceNames: []string{kubeadmconstants.GetKubeletConfigMapName(k8sVersion, legacy)},
			},
		},
	}); err != nil {
		return err
	}

	return apiclient.CreateOrUpdateRoleBinding(client, &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapRBACName(k8sVersion, legacy),
			Namespace: metav1.NamespaceSystem,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     configMapRBACName(k8sVersion, legacy),
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

// configMapRBACName returns the name for the Role/RoleBinding for the kubelet config configmap for the right branch of k8s
// TODO: Remove the legacy arg once UnversionedKubeletConfigMap graduates to GA:
// https://github.com/kubernetes/kubeadm/issues/1582
func configMapRBACName(k8sVersion *version.Version, legacy bool) string {
	if !legacy {
		return kubeadmconstants.KubeletBaseConfigMapRole
	}
	return fmt.Sprintf("%s%d.%d", kubeadmconstants.KubeletBaseConfigMapRolePrefix, k8sVersion.Major(), k8sVersion.Minor())
}

// writeConfigBytesToDisk writes a byte slice down to disk at the specific location of the kubelet config file
func writeConfigBytesToDisk(b []byte, kubeletDir string) error {
	configFile := filepath.Join(kubeletDir, kubeadmconstants.KubeletConfigurationFileName)
	fmt.Printf("[kubelet-start] Writing kubelet configuration to file %q\n", configFile)

	// creates target folder if not already exists
	if err := os.MkdirAll(kubeletDir, 0o700); err != nil {
		return errors.Wrapf(err, "failed to create directory %q", kubeletDir)
	}

	if err := ioutil.WriteFile(configFile, b, 0o644); err != nil {
		return errors.Wrapf(err, "failed to write kubelet configuration to the file %q", configFile)
	}
	return nil
}
