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

	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

const (
	// BootstrapDiscoveryClusterRoleName sets the name for the ClusterRole that allows
	// the bootstrap tokens to access the kubeadm-config ConfigMap during the node bootstrap/discovery
	// phase for additional master nodes
	BootstrapDiscoveryClusterRoleName = "kubeadm:bootstrap-discovery-kubeadm-config"
)

// UploadConfiguration saves the InitConfiguration used for later reference (when upgrading for instance)
func UploadConfiguration(cfg *kubeadmapi.InitConfiguration, client clientset.Interface) error {

	fmt.Printf("[uploadconfig] storing the configuration used in ConfigMap %q in the %q Namespace\n", kubeadmconstants.InitConfigurationConfigMap, metav1.NamespaceSystem)

	// We don't want to mutate the cfg itself, so create a copy of it using .DeepCopy of it first
	cfgToUpload := cfg.DeepCopy()
	// Removes sensitive info from the data that will be stored in the config map
	cfgToUpload.BootstrapTokens = nil
	// Clear the NodeRegistration object.
	cfgToUpload.NodeRegistration = kubeadmapi.NodeRegistrationOptions{}
	// TODO: Reset the .ComponentConfig struct like this:
	// cfgToUpload.ComponentConfigs = kubeadmapi.ComponentConfigs{}
	// in order to not upload any other components' config to the kubeadm-config
	// ConfigMap. The components store their config in their own ConfigMaps.
	// Before this line can be uncommented util/config.loadConfigurationBytes()
	// needs to support reading the different components' ConfigMaps first.

	// Marshal the object into YAML
	cfgYaml, err := configutil.MarshalKubeadmConfigObject(cfgToUpload)
	if err != nil {
		fmt.Println("err", err.Error())
		return err
	}

	err = apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.InitConfigurationConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.InitConfigurationConfigMapKey: string(cfgYaml),
		},
	})
	if err != nil {
		return err
	}

	// Ensure that the BootstrapDiscoveryClusterRole exists
	err = apiclient.CreateOrUpdateRole(client, &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BootstrapDiscoveryClusterRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		Rules: []rbac.PolicyRule{
			rbachelper.NewRule("get").Groups("").Resources("configmaps").Names(kubeadmconstants.InitConfigurationConfigMap).RuleOrDie(),
		},
	})
	if err != nil {
		return err
	}

	// Binds the BootstrapDiscoveryClusterRole to all the bootstrap tokens
	// that are members of the system:bootstrappers:kubeadm:default-node-token group
	return apiclient.CreateOrUpdateRoleBinding(client, &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BootstrapDiscoveryClusterRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     BootstrapDiscoveryClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: rbac.GroupKind,
				Name: kubeadmconstants.NodeBootstrapTokenAuthGroup,
			},
		},
	})
}
