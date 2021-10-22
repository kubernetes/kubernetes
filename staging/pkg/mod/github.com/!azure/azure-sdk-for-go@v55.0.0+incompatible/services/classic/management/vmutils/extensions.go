// +build go1.7

package vmutils

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	vm "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachine"
)

const (
	dockerPublicConfigVersion = 2
)

func AddAzureVMExtensionConfiguration(role *vm.Role, name, publisher, version, referenceName, state string,
	publicConfigurationValue, privateConfigurationValue []byte) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	extension := vm.ResourceExtensionReference{
		Name:          name,
		Publisher:     publisher,
		Version:       version,
		ReferenceName: referenceName,
		State:         state,
	}

	if len(privateConfigurationValue) != 0 {
		extension.ParameterValues = append(extension.ParameterValues, vm.ResourceExtensionParameter{
			Key:   "ignored",
			Value: base64.StdEncoding.EncodeToString(privateConfigurationValue),
			Type:  "Private",
		})
	}

	if len(publicConfigurationValue) != 0 {
		extension.ParameterValues = append(extension.ParameterValues, vm.ResourceExtensionParameter{
			Key:   "ignored",
			Value: base64.StdEncoding.EncodeToString(publicConfigurationValue),
			Type:  "Public",
		})
	}

	if role.ResourceExtensionReferences == nil {
		role.ResourceExtensionReferences = &[]vm.ResourceExtensionReference{}
	}
	extensionList := append(*role.ResourceExtensionReferences, extension)
	role.ResourceExtensionReferences = &extensionList
	return nil
}

// AddAzureDockerVMExtensionConfiguration adds the DockerExtension to the role
// configuratioon and opens a port "dockerPort"
// TODO(ahmetalpbalkan) Deprecate this and move to 'docker-machine' codebase.
func AddAzureDockerVMExtensionConfiguration(role *vm.Role, dockerPort int, version string) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	if err := ConfigureWithExternalPort(role, "docker", dockerPort, dockerPort, vm.InputEndpointProtocolTCP); err != nil {
		return err
	}

	publicConfiguration, err := createDockerPublicConfig(dockerPort)
	if err != nil {
		return err
	}

	privateConfiguration, err := json.Marshal(dockerPrivateConfig{})
	if err != nil {
		return err
	}

	return AddAzureVMExtensionConfiguration(role,
		"DockerExtension", "MSOpenTech.Extensions",
		version, "DockerExtension", "enable",
		publicConfiguration, privateConfiguration)
}

func createDockerPublicConfig(dockerPort int) ([]byte, error) {
	return json.Marshal(dockerPublicConfig{DockerPort: dockerPort, Version: dockerPublicConfigVersion})
}

type dockerPublicConfig struct {
	DockerPort int `json:"dockerport"`
	Version    int `json:"version"`
}

type dockerPrivateConfig struct{}
