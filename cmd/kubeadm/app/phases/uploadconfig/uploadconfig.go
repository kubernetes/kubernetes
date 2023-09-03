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

package uploadconfig

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

const (
	// NodesKubeadmConfigClusterRoleName sets the name for the ClusterRole that allows
	// the bootstrap tokens to access the kubeadm-config ConfigMap during the node bootstrap/discovery
	// or during upgrade nodes
	NodesKubeadmConfigClusterRoleName = "kubeadm:nodes-kubeadm-config"
)

// UploadConfiguration saves the InitConfiguration used for later reference (when upgrading for instance)
func UploadConfiguration(cfg *kubeadmapi.InitConfiguration, client clientset.Interface) error {
	fmt.Printf("[upload-config] Storing the configuration used in ConfigMap %q in the %q Namespace\n", kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)

	// Prepare the ClusterConfiguration for upload
	// The components store their config in their own ConfigMaps, then reset the .ComponentConfig struct;
	// We don't want to mutate the cfg itself, so create a copy of it using .DeepCopy of it first
	clusterConfigurationToUpload := cfg.ClusterConfiguration.DeepCopy()
	clusterConfigurationToUpload.ComponentConfigs = kubeadmapi.ComponentConfigMap{}

	// restore the resolved Kubernetes version as CI Kubernetes version if needed
	if len(clusterConfigurationToUpload.CIKubernetesVersion) > 0 {
		clusterConfigurationToUpload.KubernetesVersion = clusterConfigurationToUpload.CIKubernetesVersion
	}

	// Marshal the ClusterConfiguration into YAML
	clusterConfigurationYaml, err := configutil.MarshalKubeadmConfigObject(clusterConfigurationToUpload, kubeadmapiv1.SchemeGroupVersion)
	if err != nil {
		return err
	}

	err = apiclient.CreateOrMutateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeadmConfigConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.ClusterConfigurationConfigMapKey: string(clusterConfigurationYaml),
		},
	}, func(cm *v1.ConfigMap) error {
		// Upgrade will call to UploadConfiguration with a modified KubernetesVersion reflecting the new
		// Kubernetes version. In that case, the mutation path will take place.
		cm.Data[kubeadmconstants.ClusterConfigurationConfigMapKey] = string(clusterConfigurationYaml)
		return nil
	})
	if err != nil {
		return err
	}

	// Ensure that the NodesKubeadmConfigClusterRoleName exists
	err = apiclient.CreateOrUpdateRole(client, &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      NodesKubeadmConfigClusterRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		Rules: []rbac.PolicyRule{
			{
				Verbs:         []string{"get"},
				APIGroups:     []string{""},
				Resources:     []string{"configmaps"},
				ResourceNames: []string{kubeadmconstants.KubeadmConfigConfigMap},
			},
		},
	})
	if err != nil {
		return err
	}

	// Binds the NodesKubeadmConfigClusterRoleName to all the bootstrap tokens
	// that are members of the system:bootstrappers:kubeadm:default-node-token group
	// and to all nodes
	return apiclient.CreateOrUpdateRoleBinding(client, &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      NodesKubeadmConfigClusterRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     NodesKubeadmConfigClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: rbac.GroupKind,
				Name: kubeadmconstants.NodeBootstrapTokenAuthGroup,
			},
			{
				Kind: rbac.GroupKind,
				Name: kubeadmconstants.NodesGroup,
			},
		},
	})
}
