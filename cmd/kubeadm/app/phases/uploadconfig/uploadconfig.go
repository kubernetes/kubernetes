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
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
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

	// Get the kubeadm ConfigMap
	configMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(kubeadmconstants.KubeadmConfigConfigMap, metav1.GetOptions{})
	if err != nil {
		return errors.Wrap(err, "failed to get config map")
	}

	// Handle missing ClusterConfiguration in the ConfigMap. Should only happen if someone manually
	// interacted with the ConfigMap.
	clusterConfigurationYaml, ok := configMap.Data[kubeadmconstants.ClusterConfigurationConfigMapKey]
	if !ok {
		return errors.Errorf("cannot find key %q in ConfigMap %q in the %q Namespace",
			kubeadmconstants.ClusterConfigurationConfigMapKey, kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
	}

	// Obtain the existing ClusterStatus object
	clusterStatus, err := configutil.UnmarshalClusterStatus(configMap.Data)
	if err != nil {
		return err
	}

	// Handle a nil APIEndpoints map. Should only happen if someone manually
	// interacted with the ConfigMap.
	if clusterStatus.APIEndpoints == nil {
		return errors.Errorf("APIEndpoints from ConfigMap %q in the %q Namespace is nil",
			kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
	}

	// Check for existence of the nodeName key in the list of APIEndpoints.
	// Return early if it's missing.
	apiEndpoint, ok := clusterStatus.APIEndpoints[nodeName]
	if !ok {
		klog.Warningf("No APIEndpoint registered for node %q", nodeName)
		return nil
	}

	klog.V(2).Infof("Removing APIEndpoint %#v for node %q", apiEndpoint, nodeName)
	delete(clusterStatus.APIEndpoints, nodeName)

	// Marshal the ClusterStatus back into YAML
	clusterStatusYaml, err := configutil.MarshalKubeadmConfigObject(clusterStatus)
	if err != nil {
		return err
	}

	// Update the ClusterStatus in the ConfigMap
	return apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeadmConfigConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.ClusterConfigurationConfigMapKey: clusterConfigurationYaml,
			kubeadmconstants.ClusterStatusConfigMapKey:        string(clusterStatusYaml),
		},
	})
}

// UploadConfiguration saves the InitConfiguration used for later reference (when upgrading for instance)
func UploadConfiguration(cfg *kubeadmapi.InitConfiguration, client clientset.Interface) error {
	fmt.Printf("[upload-config] storing the configuration used in ConfigMap %q in the %q Namespace\n", kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)

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
	// Gets the current cluster status
	// TODO: use configmap locks on this object on the get before the update.
	clusterStatus, err := configutil.GetClusterStatus(client)
	if err != nil {
		return err
	}

	// Updates the ClusterStatus with the current control plane instance
	if clusterStatus.APIEndpoints == nil {
		clusterStatus.APIEndpoints = map[string]kubeadmapi.APIEndpoint{}
	}
	clusterStatus.APIEndpoints[cfg.NodeRegistration.Name] = cfg.LocalAPIEndpoint

	// Marshal the ClusterStatus back into YAML
	clusterStatusYaml, err := configutil.MarshalKubeadmConfigObject(clusterStatus)
	if err != nil {
		return err
	}

	err = apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeadmConfigConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.ClusterConfigurationConfigMapKey: string(clusterConfigurationYaml),
			kubeadmconstants.ClusterStatusConfigMapKey:        string(clusterStatusYaml),
		},
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
			rbachelper.NewRule("get").Groups("").Resources("configmaps").Names(kubeadmconstants.KubeadmConfigConfigMap).RuleOrDie(),
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
