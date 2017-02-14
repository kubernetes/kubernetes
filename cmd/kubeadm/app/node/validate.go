/*
Copyright 2016 The Kubernetes Authors.

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

package node

import (
	"fmt"

	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/pkg/apis/certificates"
)


func ValidateAPIServer(config *clientcmdapi.Config) error {
	client, err := kubeconfigutil.KubeConfigToClientSet(config)
	if err != nil {
		return err
	}

	version, err := client.DiscoveryClient.ServerVersion()
	if err != nil {
		return fmt.Errorf("failed to check server version: %v", err)
	}
	fmt.Printf("[bootstrap] Detected server version: %s\n", version.String())

	// check certificates API
	serverGroups, err := client.DiscoveryClient.ServerGroups()
	if err != nil {
		return fmt.Errorf("certificate API check failed: failed to retrieve a list of supported API objects [%v]", err)
	}
	for _, group := range serverGroups.Groups {
		if group.Name == certificates.GroupName {
			return nil
		}
	}
	return fmt.Errorf("certificate API check failed: API version %s does not support certificates API, use v1.4.0 or newer", version.String())
}
