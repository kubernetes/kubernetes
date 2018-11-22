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
	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

// WriteConfigToDisk writes the kubelet config object down to a file
// Used at "kubeadm init" and "kubeadm upgrade" time
func WriteConfigToDisk(kubeletConfig *kubeletconfig.KubeletConfiguration, kubeletDir string) error {

	kubeletBytes, err := getConfigBytes(kubeletConfig)
	if err != nil {
		return err
	}
	return writeConfigBytesToDisk(kubeletBytes, kubeletDir)
}

// CreateConfigMap creates a ConfigMap with the generic kubelet configuration.
// Used at "kubeadm init" and "kubeadm upgrade" time
func CreateConfigMap(cfg *kubeadmapi.InitConfiguration, client clientset.Interface) error {

	k8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return err
	}

	configMapName := kubeadmconstants.GetKubeletConfigMapName(k8sVersion)
	fmt.Printf("[kubelet] Creating a ConfigMap %q in namespace %s with the configuration for the kubelets in the cluster\n", configMapName, metav1.NamespaceSystem)

	kubeletBytes, err := getConfigBytes(cfg.ComponentConfigs.Kubelet)
	if err != nil {
		return err
	}

	if err := apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.KubeletBaseConfigurationConfigMapKey: string(kubeletBytes),
		},
	}); err != nil {
		return err
	}

	if err := createConfigMapRBACRules(client, k8sVersion); err != nil {
		return errors.Wrap(err, "error creating kubelet configuration configmap RBAC rules")
	}
	return nil
}

// createConfigMapRBACRules creates the RBAC rules for exposing the base kubelet ConfigMap in the kube-system namespace to unauthenticated users
func createConfigMapRBACRules(client clientset.Interface, k8sVersion *version.Version) error {
	if err := apiclient.CreateOrUpdateRole(client, &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapRBACName(k8sVersion),
			Namespace: metav1.NamespaceSystem,
		},
		Rules: []rbac.PolicyRule{
			rbachelper.NewRule("get").Groups("").Resources("configmaps").Names(kubeadmconstants.GetKubeletConfigMapName(k8sVersion)).RuleOrDie(),
		},
	}); err != nil {
		return err
	}

	return apiclient.CreateOrUpdateRoleBinding(client, &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapRBACName(k8sVersion),
			Namespace: metav1.NamespaceSystem,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     configMapRBACName(k8sVersion),
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

// DownloadConfig downloads the kubelet configuration from a ConfigMap and writes it to disk.
// Used at "kubeadm join" time
func DownloadConfig(client clientset.Interface, kubeletVersion *version.Version, kubeletDir string) error {

	// Download the ConfigMap from the cluster based on what version the kubelet is
	configMapName := kubeadmconstants.GetKubeletConfigMapName(kubeletVersion)

	fmt.Printf("[kubelet] Downloading configuration for the kubelet from the %q ConfigMap in the %s namespace\n",
		configMapName, metav1.NamespaceSystem)

	kubeletCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(configMapName, metav1.GetOptions{})
	// If the ConfigMap wasn't found and the kubelet version is v1.10.x, where we didn't support the config file yet
	// just return, don't error out
	if apierrors.IsNotFound(err) && kubeletVersion.Minor() == 10 {
		return nil
	}
	if err != nil {
		return err
	}

	return writeConfigBytesToDisk([]byte(kubeletCfg.Data[kubeadmconstants.KubeletBaseConfigurationConfigMapKey]), kubeletDir)
}

// configMapRBACName returns the name for the Role/RoleBinding for the kubelet config configmap for the right branch of k8s
func configMapRBACName(k8sVersion *version.Version) string {
	return fmt.Sprintf("%s%d.%d", kubeadmconstants.KubeletBaseConfigMapRolePrefix, k8sVersion.Major(), k8sVersion.Minor())
}

// getConfigBytes marshals a KubeletConfiguration object to bytes
func getConfigBytes(kubeletConfig *kubeletconfig.KubeletConfiguration) ([]byte, error) {
	return componentconfigs.Known[componentconfigs.KubeletConfigurationKind].Marshal(kubeletConfig)
}

// writeConfigBytesToDisk writes a byte slice down to disk at the specific location of the kubelet config file
func writeConfigBytesToDisk(b []byte, kubeletDir string) error {
	configFile := filepath.Join(kubeletDir, kubeadmconstants.KubeletConfigurationFileName)
	fmt.Printf("[kubelet-start] Writing kubelet configuration to file %q\n", configFile)

	// creates target folder if not already exists
	if err := os.MkdirAll(kubeletDir, 0700); err != nil {
		return errors.Wrapf(err, "failed to create directory %q", kubeletDir)
	}

	if err := ioutil.WriteFile(configFile, b, 0644); err != nil {
		return errors.Wrapf(err, "failed to write kubelet configuration to the file %q", configFile)
	}
	return nil
}
