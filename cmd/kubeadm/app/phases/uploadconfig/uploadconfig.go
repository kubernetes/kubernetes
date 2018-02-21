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

	"github.com/ghodss/yaml"

	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

const (
	// BootstrapDiscoveryClusterRoleName sets the name for the ClusterRole that allows
	// the bootstrap tokens to access the kubeadm-config ConfigMap during the node bootstrap/discovery
	// phase for additional master nodes
	BootstrapDiscoveryClusterRoleName = "kubeadm:bootstrap-discovery-kubeadm-config"
)

// UploadConfiguration saves the MasterConfiguration used for later reference (when upgrading for instance)
func UploadConfiguration(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {

	fmt.Printf("[uploadconfig] Storing the configuration used in ConfigMap %q in the %q Namespace\n", kubeadmconstants.MasterConfigurationConfigMap, metav1.NamespaceSystem)

	// Convert cfg to the external version as that's the only version of the API that can be deserialized later
	externalcfg := &kubeadmapiext.MasterConfiguration{}
	legacyscheme.Scheme.Convert(cfg, externalcfg, nil)

	// Removes sensitive info from the data that will be stored in the config map
	externalcfg.Token = ""

	cfgYaml, err := yaml.Marshal(*externalcfg)
	if err != nil {
		return err
	}

	err = apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.MasterConfigurationConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.MasterConfigurationConfigMapKey: string(cfgYaml),
		},
	})
	if err != nil {
		return err
	}

	// Ensure that the BootstrapDiscoveryClusterRole exists and allows
	// access to kubeadm-config ConfigMap during the node bootstrap/discovery phase
	err = apiclient.CreateOrUpdateRole(client, &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BootstrapDiscoveryClusterRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		Rules: []rbac.PolicyRule{
			rbachelper.NewRule("get").Groups("").Resources("configmaps").Names(kubeadmconstants.MasterConfigurationConfigMap).RuleOrDie(),
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
