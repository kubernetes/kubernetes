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
	"k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/pkg/api"
)

// FetchConfiguration fetches configuration required for upgrading your cluster from a file (which has precedence) or a ConfigMap in the cluster
func FetchConfiguration(client clientset.Interface, w io.Writer, cfgPath string) (*kubeadmapiext.MasterConfiguration, error) {
	fmt.Println("[upgrade/config] Making sure the configuration is correct:")

	// Load the configuration from a file or the cluster
	configBytes, err := loadConfigurationBytes(client, w, cfgPath)
	if err != nil {
		return nil, err
	}

	// Take the versioned configuration populated from the configmap, default it and validate
	// Return the internal version of the API object
	versionedcfg, err := bytesToValidatedMasterConfig(configBytes)
	if err != nil {
		return nil, fmt.Errorf("could not decode configuration: %v", err)
	}
	return versionedcfg, nil
}

// loadConfigurationBytes loads the configuration byte slice from either a file or the cluster ConfigMap
func loadConfigurationBytes(client clientset.Interface, w io.Writer, cfgPath string) ([]byte, error) {
	if cfgPath != "" {
		fmt.Printf("[upgrade/config] Reading configuration options from a file: %s\n", cfgPath)
		return ioutil.ReadFile(cfgPath)
	}

	fmt.Println("[upgrade/config] Reading configuration from the cluster...")

	configMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(constants.MasterConfigurationConfigMap, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		fmt.Printf("[upgrade/config] In order to upgrade, a ConfigMap called %q in the %s namespace must exist.\n", constants.MasterConfigurationConfigMap, metav1.NamespaceSystem)
		fmt.Println("[upgrade/config] Without this information, 'kubeadm upgrade' don't how to configure your upgraded cluster.")
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

// bytesToValidatedMasterConfig converts a byte array to an external, defaulted and validated configuration object
func bytesToValidatedMasterConfig(b []byte) (*kubeadmapiext.MasterConfiguration, error) {
	cfg := &kubeadmapiext.MasterConfiguration{}
	finalCfg := &kubeadmapiext.MasterConfiguration{}
	internalcfg := &kubeadmapi.MasterConfiguration{}

	if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), b, cfg); err != nil {
		return nil, fmt.Errorf("unable to decode config from bytes: %v", err)
	}
	// Default and convert to the internal version
	api.Scheme.Default(cfg)
	api.Scheme.Convert(cfg, internalcfg, nil)

	// Applies dynamic defaults to settings not provided with flags
	if err := configutil.SetInitDynamicDefaults(internalcfg); err != nil {
		return nil, err
	}
	// Validates cfg (flags/configs + defaults + dynamic defaults)
	if err := validation.ValidateMasterConfiguration(internalcfg).ToAggregate(); err != nil {
		return nil, err
	}
	// Finally converts back to the external version
	api.Scheme.Convert(internalcfg, finalCfg, nil)
	return finalCfg, nil
}
