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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// UploadConfiguration saves the MasterConfiguration used for later reference (when upgrading for instance)
func UploadConfiguration(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {

	fmt.Printf("[uploadconfig] storing the configuration used in ConfigMap %q in the %q Namespace\n", kubeadmconstants.MasterConfigurationConfigMap, metav1.NamespaceSystem)

	// Convert cfg to the external version as that's the only version of the API that can be deserialized later
	externalcfg := &kubeadmapiv1alpha2.MasterConfiguration{}
	kubeadmscheme.Scheme.Convert(cfg, externalcfg, nil)

	// Removes sensitive info from the data that will be stored in the config map
	externalcfg.BootstrapTokens = nil
	// Clear the NodeRegistration object.
	externalcfg.NodeRegistration = kubeadmapiv1alpha2.NodeRegistrationOptions{}

	cfgYaml, err := util.MarshalToYamlForCodecs(externalcfg, kubeadmapiv1alpha2.SchemeGroupVersion, scheme.Codecs)
	if err != nil {
		return err
	}

	return apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.MasterConfigurationConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			kubeadmconstants.MasterConfigurationConfigMapKey: string(cfgYaml),
		},
	})
}
