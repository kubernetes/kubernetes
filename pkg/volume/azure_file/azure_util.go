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
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	fileMode        = "file_mode"
	dirMode         = "dir_mode"
	gid             = "gid"
	vers            = "vers"
	defaultFileMode = "0777"
	defaultDirMode  = "0777"
	defaultVers     = "3.0"
)

// Abstract interface to azure file operations.
type azureUtil interface {
	GetAzureCredentials(host volume.VolumeHost, nameSpace, secretName string) (string, string, error)
	SetAzureCredentials(host volume.VolumeHost, nameSpace, accountName, accountKey string) (string, error)
}

type azureSvc struct{}

func (s *azureSvc) GetAzureCredentials(host volume.VolumeHost, nameSpace, secretName string) (string, string, error) {
	var accountKey, accountName string
	kubeClient := host.GetKubeClient()
	if kubeClient == nil {
		return "", "", fmt.Errorf("Cannot get kube client")
	}

	keys, err := kubeClient.CoreV1().Secrets(nameSpace).Get(secretName, metav1.GetOptions{})
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

func (s *azureSvc) SetAzureCredentials(host volume.VolumeHost, nameSpace, accountName, accountKey string) (string, error) {
	kubeClient := host.GetKubeClient()
	if kubeClient == nil {
		return "", fmt.Errorf("Cannot get kube client")
	}
	secretName := "azure-storage-account-" + accountName + "-secret"
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: nameSpace,
			Name:      secretName,
		},
		Data: map[string][]byte{
			"azurestorageaccountname": []byte(accountName),
			"azurestorageaccountkey":  []byte(accountKey),
		},
		Type: "Opaque",
	}
	_, err := kubeClient.CoreV1().Secrets(nameSpace).Create(secret)
	if errors.IsAlreadyExists(err) {
		err = nil
	}
	if err != nil {
		return "", fmt.Errorf("Couldn't create secret %v", err)
	}
	return secretName, err
}

// check whether mountOptions contain file_mode, dir_mode, vers, gid, if not, append default mode
func appendDefaultMountOptions(mountOptions []string, fsGroup *int64) []string {
	fileModeFlag := false
	dirModeFlag := false
	versFlag := false
	gidFlag := false

	for _, mountOption := range mountOptions {
		if strings.HasPrefix(mountOption, fileMode) {
			fileModeFlag = true
		}
		if strings.HasPrefix(mountOption, dirMode) {
			dirModeFlag = true
		}
		if strings.HasPrefix(mountOption, vers) {
			versFlag = true
		}
		if strings.HasPrefix(mountOption, gid) {
			gidFlag = true
		}
	}

	allMountOptions := mountOptions
	if !fileModeFlag {
		allMountOptions = append(allMountOptions, fmt.Sprintf("%s=%s", fileMode, defaultFileMode))
	}

	if !dirModeFlag {
		allMountOptions = append(allMountOptions, fmt.Sprintf("%s=%s", dirMode, defaultDirMode))
	}

	if !versFlag {
		allMountOptions = append(allMountOptions, fmt.Sprintf("%s=%s", vers, defaultVers))
	}

	if !gidFlag && fsGroup != nil {
		allMountOptions = append(allMountOptions, fmt.Sprintf("%s=%d", gid, *fsGroup))
	}
	return allMountOptions
}
