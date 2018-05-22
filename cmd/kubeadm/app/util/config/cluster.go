/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"io"
	"io/ioutil"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// TODO: Add unit tests for this file

// FetchConfigFromFileOrCluster fetches configuration required for upgrading your cluster from a file (which has precedence) or a ConfigMap in the cluster
func FetchConfigFromFileOrCluster(client clientset.Interface, w io.Writer, logPrefix, cfgPath string) (*kubeadmapi.MasterConfiguration, error) {
	// Load the configuration from a file or the cluster
	configBytes, err := loadConfigurationBytes(client, w, logPrefix, cfgPath)
	if err != nil {
		return nil, err
	}

	// Take the versioned configuration populated from the file or ConfigMap, convert it to internal, default and validate
	return BytesToInternalConfig(configBytes)
}

// loadConfigurationBytes loads the configuration byte slice from either a file or the cluster ConfigMap
func loadConfigurationBytes(client clientset.Interface, w io.Writer, logPrefix, cfgPath string) ([]byte, error) {
	// The config file has the highest priority
	if cfgPath != "" {
		fmt.Fprintf(w, "[%s] Reading configuration options from a file: %s\n", logPrefix, cfgPath)
		return ioutil.ReadFile(cfgPath)
	}

	fmt.Fprintf(w, "[%s] Reading configuration from the cluster...\n", logPrefix)

	configMap, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(constants.MasterConfigurationConfigMap, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		// Return the apierror directly so the caller of this function can know what type of error occurred and act based on that
		return []byte{}, err
	} else if err != nil {
		return []byte{}, fmt.Errorf("an unexpected error happened when trying to get the ConfigMap %q in the %s namespace: %v", constants.MasterConfigurationConfigMap, metav1.NamespaceSystem, err)
	}

	fmt.Fprintf(w, "[%s] FYI: You can look at this config file with 'kubectl -n %s get cm %s -oyaml'\n", logPrefix, metav1.NamespaceSystem, constants.MasterConfigurationConfigMap)
	return []byte(configMap.Data[constants.MasterConfigurationConfigMapKey]), nil
}
