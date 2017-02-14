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

package token

import (
	"fmt"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/controller/bootstrap"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const BootstrapKubeConfigContext = "bootstrap-context"

func CreateBootstrapConfigMap(file string) error {
	adminConfig, err := clientcmd.LoadFromFile(file)
	if err != nil {
		return fmt.Errorf("failed to load admin kubeconfig [%v]", err)
	}
	client, err := kubeconfigutil.KubeConfigToClientSet(adminConfig)
	if err != nil {
		return err
	}

	adminCluster := adminConfig.Contexts[adminConfig.CurrentContext].Cluster
	// Copy the cluster from admin.conf to the bootstrap kubeconfig, contains the CA cert and the server URL
	bootstrapConfig := &clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			adminCluster: adminConfig.Clusters[adminCluster],
		},
		Contexts: map[string]*clientcmdapi.Context{
			BootstrapKubeConfigContext: &clientcmdapi.Context{
				Cluster: adminCluster,
			},
		},
		CurrentContext: BootstrapKubeConfigContext,
	}
	bootstrapBytes, err := clientcmd.Write(*bootstrapConfig)
	if err != nil {
		return err
	}

	bootstrapConfigMap := v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: bootstrap.ConfigMapClusterInfo},
		Data: map[string]string{
			bootstrap.KubeConfigKey: string(bootstrapBytes),
		},
	}

	if _, err := client.CoreV1().ConfigMaps(metav1.NamespacePublic).Create(&bootstrapConfigMap); err != nil {
		return err
	}
	return nil
}
