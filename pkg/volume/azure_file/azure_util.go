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

package azure_file

import (
	"fmt"

	"k8s.io/kubernetes/pkg/volume"
)

// Abstract interface to azure file operations.
type azureUtil interface {
	GetAzureCredentials(host volume.VolumeHost, nameSpace, secretName, shareName string) (string, string, error)
}

type azureSvc struct{}

func (s *azureSvc) GetAzureCredentials(host volume.VolumeHost, nameSpace, secretName, shareName string) (string, string, error) {
	var accountKey, accountName string
	kubeClient := host.GetKubeClient()
	if kubeClient == nil {
		return "", "", fmt.Errorf("Cannot get kube client")
	}

	keys, err := kubeClient.Core().Secrets(nameSpace).Get(secretName)
	if err != nil {
		return "", "", fmt.Errorf("Couldn't get secret %v/%v", nameSpace, secretName)
	}
	for name, data := range keys.Data {
		if name == "azurestorageaccountname" {
			accountName = string(data)
		}
		if name == "azurestorageaccountkey" {
			accountKey = string(data)
		}
	}
	if accountName == "" || accountKey == "" {
		return "", "", fmt.Errorf("Invalid %v/%v, couldn't extract azurestorageaccountname or azurestorageaccountkey", nameSpace, secretName)
	}
	return accountName, accountKey, nil
}
