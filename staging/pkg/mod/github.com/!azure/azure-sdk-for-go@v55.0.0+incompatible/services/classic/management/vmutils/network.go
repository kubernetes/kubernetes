// +build go1.7

package vmutils

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"fmt"

	vm "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachine"
)

// ConfigureWithPublicSSH adds configuration exposing port 22 externally
func ConfigureWithPublicSSH(role *vm.Role) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	return ConfigureWithExternalPort(role, "SSH", 22, 22, vm.InputEndpointProtocolTCP)
}

// ConfigureWithPublicRDP adds configuration exposing port 3389 externally
func ConfigureWithPublicRDP(role *vm.Role) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	return ConfigureWithExternalPort(role, "RDP", 3389, 3389, vm.InputEndpointProtocolTCP)
}

// ConfigureWithPublicPowerShell adds configuration exposing port 5986
// externally
func ConfigureWithPublicPowerShell(role *vm.Role) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	return ConfigureWithExternalPort(role, "PowerShell", 5986, 5986, vm.InputEndpointProtocolTCP)
}

// ConfigureWithExternalPort adds a new InputEndpoint to the Role, exposing a
// port externally
func ConfigureWithExternalPort(role *vm.Role, name string, localport, externalport int, protocol vm.InputEndpointProtocol) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	role.ConfigurationSets = updateOrAddConfig(role.ConfigurationSets, vm.ConfigurationSetTypeNetwork,
		func(config *vm.ConfigurationSet) {
			config.InputEndpoints = append(config.InputEndpoints, vm.InputEndpoint{
				LocalPort: localport,
				Name:      name,
				Port:      externalport,
				Protocol:  protocol,
			})
		})

	return nil
}

// ConfigureWithSecurityGroup associates the Role with a specific network security group
func ConfigureWithSecurityGroup(role *vm.Role, networkSecurityGroup string) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	role.ConfigurationSets = updateOrAddConfig(role.ConfigurationSets, vm.ConfigurationSetTypeNetwork,
		func(config *vm.ConfigurationSet) {
			config.NetworkSecurityGroup = networkSecurityGroup
		})

	return nil
}

// ConfigureWithSubnet associates the Role with a specific subnet
func ConfigureWithSubnet(role *vm.Role, subnet string) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	role.ConfigurationSets = updateOrAddConfig(role.ConfigurationSets, vm.ConfigurationSetTypeNetwork,
		func(config *vm.ConfigurationSet) {
			config.SubnetNames = append(config.SubnetNames, subnet)
		})

	return nil
}
