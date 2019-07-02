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

	"github.com/pkg/errors"
	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
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

// ResetClusterStatusForNode removes the APIEndpoint of a given control-plane node
// from the ClusterStatus and updates the kubeadm ConfigMap
func ResetClusterStatusForNode(nodeName string, client clientset.Interface) error {
	fmt.Printf("[reset] Removing info for node %q from the ConfigMap %q in the %q Namespace\n",
		nodeName, kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)

	return apiclient.MutateConfigMap(client, metav1.ObjectMeta{
		Name:      kubeadmconstants.KubeadmConfigConfigMap,
		Namespace: metav1.NamespaceSystem,
	}, func(cm *v1.ConfigMap) error {
		return mutateClusterStatus(cm, func(cs *kubeadmapi.ClusterStatus) error {
			// Handle a nil APIEndpoints map. Should only happen if someone manually
			// interacted with the ConfigMap.
			if cs.APIEndpoints == nil {
				return errors.Errorf("APIEndpoints from ConfigMap %q in the %q Namespace is nil",
					kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
			}
			klog.V(2).Infof("Removing APIEndpoint for Node %q", nodeName)
			delete(cs.APIEndpoints, nodeName)
			return nil
		})
	})
}

// UploadConfiguration saves the InitConfiguration used for later reference (when upgrading for instance)
func UploadConfiguration(cfg *kubeadmapi.InitConfiguration, client clientset.Interface) error {
	fmt.Printf("[upload-config] Storing the configuration used in ConfigMap %q in the %q Namespace\n", kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)

	// Prepare the ClusterConfiguration for upload
	// The components store their config in their own ConfigMaps, then reset the .ComponentConfig struct;
	// We don't want to mutate the cfg itself, so create a copy of it using .DeepCopy of it first
	clusterConfigurationToUpload := cfg.ClusterConfiguration.DeepCopy()
	clusterConfigurationToUpload.ComponentConfigs = kubeadmapi.ComponentConfigs{}

	// Marshal the ClusterConfiguration into YAML
	clusterConfigurationYaml, err := configutil.MarshalKubeadmConfigObject(clusterConfigurationToUpload)
	if err != nil {
		return err
	}

	// Prepare the ClusterStatus for upload
	clusterStatus := &kubeadmapi.ClusterStatus{
		APIEndpoints: map[string]kubeadmapi.APIEndpoint{
			cfg.NodeRegistration.Name: cfg.LocalAPIEndpoint,
		},
	}
	// Marshal the ClusterStatus into YAML
	clusterStatusYaml, err := configutil.MarshalKubeadmConfigObject(clusterStatus)
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
			kubeadmconstants.ClusterStatusConfigMapKey:        string(clusterStatusYaml),
		},
	}, func(cm *v1.ConfigMap) error {
		// Upgrade will call to UploadConfiguration with a modified KubernetesVersion reflecting the new
		// Kubernetes version. In that case, the mutation path will take place.
		cm.Data[kubeadmconstants.ClusterConfigurationConfigMapKey] = string(clusterConfigurationYaml)
		// Mutate the ClusterStatus now
		return mutateClusterStatus(cm, func(cs *kubeadmapi.ClusterStatus) error {
			// Handle a nil APIEndpoints map. Should only happen if someone manually
			// interacted with the ConfigMap.
			if cs.APIEndpoints == nil {
				return errors.Errorf("APIEndpoints from ConfigMap %q in the %q Namespace is nil",
					kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
			}
			cs.APIEndpoints[cfg.NodeRegistration.Name] = cfg.LocalAPIEndpoint
			return nil
		})
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

func mutateClusterStatus(cm *v1.ConfigMap, mutator func(*kubeadmapi.ClusterStatus) error) error {
	// Obtain the existing ClusterStatus object
	clusterStatus, err := configutil.UnmarshalClusterStatus(cm.Data)
	if err != nil {
		return err
	}
	// Mutate the ClusterStatus
	if err := mutator(clusterStatus); err != nil {
		return err
	}
	// Marshal the ClusterStatus back into YAML
	clusterStatusYaml, err := configutil.MarshalKubeadmConfigObject(clusterStatus)
	if err != nil {
		return err
	}
	// Write the marshaled mutated cluster status back to the ConfigMap
	cm.Data[kubeadmconstants.ClusterStatusConfigMapKey] = string(clusterStatusYaml)
	return nil
}
