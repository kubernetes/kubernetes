/*
Copyright 2019 The Kubernetes Authors.

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

package installer

import (
	"fmt"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"

	addons "sigs.k8s.io/addon-operators/installer/pkg/apis/config"
)

// DownloadConfiguration pulls the previous AddonInstallerConfiguration from the cluster
func DownloadConfiguration(client clientset.Interface, version *version.Version) (*addons.AddonInstallerConfiguration, error) {
	obj, err := componentconfigs.Known[componentconfigs.AddonInstallerConfigurationKind].GetFromConfigMap(client, version)
	if err != nil || obj == nil {
		return nil, err
	}

	addonCfg, ok := obj.(*addons.AddonInstallerConfiguration)
	if !ok {
		return addonCfg, fmt.Errorf("Object decoded from %q configmap was not an %s", kubeadmconstants.AddonInstallerConfigMap, componentconfigs.AddonInstallerConfigurationKind)
	}

	fmt.Printf("[addon installer] Previous config found in ConfigMap %q, Namespace %q\n", kubeadmconstants.AddonInstallerConfigMap, metav1.NamespaceSystem)
	return addonCfg, nil
}

// uploadConfiguration saves the AddonInstallerConfiguration used for later reference (when upgrading for instance)
func uploadConfiguration(addonCfg *addons.AddonInstallerConfiguration, client clientset.Interface) error {
	fmt.Printf("[addon installer] Storing the configuration used in ConfigMap %q in the %q Namespace\n", kubeadmconstants.AddonInstallerConfigMap, metav1.NamespaceSystem)

	AddonInstallerYaml, err := componentconfigs.Known[componentconfigs.AddonInstallerConfigurationKind].Marshal(addonCfg)
	if err != nil {
		return errors.Wrap(err, "error when marshaling")
	}

	return apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.AddonInstallerConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.AddonInstallerConfigMapKey: string(AddonInstallerYaml),
		},
	})
}
