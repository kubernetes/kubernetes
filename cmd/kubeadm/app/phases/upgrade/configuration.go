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

package upgrade

import (
	"fmt"
	"io"
	"io/ioutil"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

// FetchConfiguration fetches configuration required for upgrading your cluster from a file (which has precedence) or a ConfigMap in the cluster
func FetchConfiguration(client clientset.Interface, w io.Writer, cfgPath string) (*kubeadmapi.MasterConfiguration, error) {
	fmt.Println("[upgrade/config] Making sure the configuration is correct:")

	// Load the configuration from a file or the cluster
	configBytes, err := loadConfigurationBytes(client, w, cfgPath)
	if err != nil {
		return nil, err
	}

	// Take the versioned configuration populated from the file or configmap, convert it to internal, default and validate
	versionedcfg, err := configutil.BytesToInternalConfig(configBytes)
	if err != nil {
		return nil, fmt.Errorf("could not decode configuration: %v", err)
	}
	return versionedcfg, nil
}

// loadConfigurationBytes loads the configuration byte slice from either a file or the cluster ConfigMap
func loadConfigurationBytes(client clientset.Interface, w io.Writer, cfgPath string) ([]byte, error) {
	// The config file has the highest priority
	if cfgPath != "" {
		fmt.Printf("[upgrade/config] Reading configuration options from a file: %s\n", cfgPath)
		return ioutil.ReadFile(cfgPath)
	}

	fmt.Println("[upgrade/config] Reading configuration from the cluster...")

	configMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(constants.MasterConfigurationConfigMap, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		fmt.Printf("[upgrade/config] In order to upgrade, a ConfigMap called %q in the %s namespace must exist.\n", constants.MasterConfigurationConfigMap, metav1.NamespaceSystem)
		fmt.Println("[upgrade/config] Without this information, 'kubeadm upgrade' won't know how to configure your upgraded cluster.")
		fmt.Println("")
		fmt.Println("[upgrade/config] Next steps:")
		fmt.Printf("\t- OPTION 1: Run 'kubeadm config upload from-flags' and specify the same CLI arguments you passed to 'kubeadm init' when you created your master.\n")
		fmt.Printf("\t- OPTION 2: Run 'kubeadm config upload from-file' and specify the same config file you passed to 'kubeadm init' when you created your master.\n")
		fmt.Printf("\t- OPTION 3: Pass a config file to 'kubeadm upgrade' using the --config flag.\n")
		fmt.Println("")
		return []byte{}, fmt.Errorf("the ConfigMap %q in the %s namespace used for getting configuration information was not found", constants.MasterConfigurationConfigMap, metav1.NamespaceSystem)
	} else if err != nil {
		return []byte{}, fmt.Errorf("an unexpected error happened when trying to get the ConfigMap %q in the %s namespace: %v", constants.MasterConfigurationConfigMap, metav1.NamespaceSystem, err)
	}

	fmt.Printf("[upgrade/config] FYI: You can look at this config file with 'kubectl -n %s get cm %s -oyaml'\n", metav1.NamespaceSystem, constants.MasterConfigurationConfigMap)
	return []byte(configMap.Data[constants.MasterConfigurationConfigMapKey]), nil
}
