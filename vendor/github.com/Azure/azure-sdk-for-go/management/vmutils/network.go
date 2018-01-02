// +build go1.7

package vmutils

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"fmt"

	vm "github.com/Azure/azure-sdk-for-go/management/virtualmachine"
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
