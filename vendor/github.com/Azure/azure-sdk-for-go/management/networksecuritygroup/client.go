// +build go1.7

// Package networksecuritygroup provides a client for Network Security Groups.
package networksecuritygroup

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
	"encoding/xml"
	"fmt"

	"github.com/Azure/azure-sdk-for-go/management"
)

const (
	createSecurityGroupURL           = "services/networking/networksecuritygroups"
	deleteSecurityGroupURL           = "services/networking/networksecuritygroups/%s"
	getSecurityGroupURL              = "services/networking/networksecuritygroups/%s?detaillevel=full"
	listSecurityGroupsURL            = "services/networking/networksecuritygroups"
	addSecurityGroupToSubnetURL      = "services/networking/virtualnetwork/%s/subnets/%s/networksecuritygroups"
	getSecurityGroupForSubnetURL     = "services/networking/virtualnetwork/%s/subnets/%s/networksecuritygroups"
	removeSecurityGroupFromSubnetURL = "services/networking/virtualnetwork/%s/subnets/%s/networksecuritygroups/%s"
	setSecurityGroupRuleURL          = "services/networking/networksecuritygroups/%s/rules/%s"
	deleteSecurityGroupRuleURL       = "services/networking/networksecuritygroups/%s/rules/%s"

	errParamNotSpecified = "Parameter %s is not specified."
)

// NewClient is used to instantiate a new SecurityGroupClient from an Azure client
func NewClient(client management.Client) SecurityGroupClient {
	return SecurityGroupClient{client: client}
}

// CreateNetworkSecurityGroup creates a new network security group within
// the context of the specified subscription
//
// https://msdn.microsoft.com/en-us/library/azure/dn913818.aspx
func (sg SecurityGroupClient) CreateNetworkSecurityGroup(
	name string,
	label string,
	location string) (management.OperationID, error) {
	if name == "" {
		return "", fmt.Errorf(errParamNotSpecified, "name")
	}
	if location == "" {
		return "", fmt.Errorf(errParamNotSpecified, "location")
	}

	data, err := xml.Marshal(SecurityGroupRequest{
		Name:     name,
		Label:    label,
		Location: location,
	})
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(createSecurityGroupURL)
	return sg.client.SendAzurePostRequest(requestURL, data)
}

// DeleteNetworkSecurityGroup deletes the specified network security group from the subscription
//
// https://msdn.microsoft.com/en-us/library/azure/dn913825.aspx
func (sg SecurityGroupClient) DeleteNetworkSecurityGroup(
	name string) (management.OperationID, error) {
	if name == "" {
		return "", fmt.Errorf(errParamNotSpecified, "name")
	}

	requestURL := fmt.Sprintf(deleteSecurityGroupURL, name)
	return sg.client.SendAzureDeleteRequest(requestURL)
}

// GetNetworkSecurityGroup returns information about the specified network security group
//
// https://msdn.microsoft.com/en-us/library/azure/dn913821.aspx
func (sg SecurityGroupClient) GetNetworkSecurityGroup(name string) (SecurityGroupResponse, error) {
	if name == "" {
		return SecurityGroupResponse{}, fmt.Errorf(errParamNotSpecified, "name")
	}

	var securityGroup SecurityGroupResponse

	requestURL := fmt.Sprintf(getSecurityGroupURL, name)
	response, err := sg.client.SendAzureGetRequest(requestURL)
	if err != nil {
		return securityGroup, err
	}

	err = xml.Unmarshal(response, &securityGroup)
	return securityGroup, err
}

// ListNetworkSecurityGroups returns a list of the network security groups
// in the specified subscription
//
// https://msdn.microsoft.com/en-us/library/azure/dn913815.aspx
func (sg SecurityGroupClient) ListNetworkSecurityGroups() (SecurityGroupList, error) {
	var securityGroups SecurityGroupList

	response, err := sg.client.SendAzureGetRequest(listSecurityGroupsURL)
	if err != nil {
		return securityGroups, err
	}

	err = xml.Unmarshal(response, &securityGroups)
	return securityGroups, err
}

// AddNetworkSecurityToSubnet associates the network security group with
// specified subnet in a virtual network
//
// https://msdn.microsoft.com/en-us/library/azure/dn913822.aspx
func (sg SecurityGroupClient) AddNetworkSecurityToSubnet(
	name string,
	subnet string,
	virtualNetwork string) (management.OperationID, error) {
	if name == "" {
		return "", fmt.Errorf(errParamNotSpecified, "name")
	}
	if subnet == "" {
		return "", fmt.Errorf(errParamNotSpecified, "subnet")
	}
	if virtualNetwork == "" {
		return "", fmt.Errorf(errParamNotSpecified, "virtualNetwork")
	}

	data, err := xml.Marshal(SecurityGroupRequest{Name: name})
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(addSecurityGroupToSubnetURL, virtualNetwork, subnet)
	return sg.client.SendAzurePostRequest(requestURL, data)
}

// GetNetworkSecurityGroupForSubnet returns information about the network
// security group associated with a subnet
//
// https://msdn.microsoft.com/en-us/library/azure/dn913817.aspx
func (sg SecurityGroupClient) GetNetworkSecurityGroupForSubnet(
	subnet string,
	virtualNetwork string) (SecurityGroupResponse, error) {
	if subnet == "" {
		return SecurityGroupResponse{}, fmt.Errorf(errParamNotSpecified, "subnet")
	}
	if virtualNetwork == "" {
		return SecurityGroupResponse{}, fmt.Errorf(errParamNotSpecified, "virtualNetwork")
	}

	var securityGroup SecurityGroupResponse

	requestURL := fmt.Sprintf(getSecurityGroupForSubnetURL, virtualNetwork, subnet)
	response, err := sg.client.SendAzureGetRequest(requestURL)
	if err != nil {
		return securityGroup, err
	}

	err = xml.Unmarshal(response, &securityGroup)
	return securityGroup, err
}

// RemoveNetworkSecurityGroupFromSubnet removes the association of the
// specified network security group from the specified subnet
//
// https://msdn.microsoft.com/en-us/library/azure/dn913820.aspx
func (sg SecurityGroupClient) RemoveNetworkSecurityGroupFromSubnet(
	name string,
	subnet string,
	virtualNetwork string) (management.OperationID, error) {
	if name == "" {
		return "", fmt.Errorf(errParamNotSpecified, "name")
	}
	if subnet == "" {
		return "", fmt.Errorf(errParamNotSpecified, "subnet")
	}
	if virtualNetwork == "" {
		return "", fmt.Errorf(errParamNotSpecified, "virtualNetwork")
	}

	requestURL := fmt.Sprintf(removeSecurityGroupFromSubnetURL, virtualNetwork, subnet, name)
	return sg.client.SendAzureDeleteRequest(requestURL)
}

// SetNetworkSecurityGroupRule adds or updates a network security rule that
// is associated with the specified network security group
//
// https://msdn.microsoft.com/en-us/library/azure/dn913819.aspx
func (sg SecurityGroupClient) SetNetworkSecurityGroupRule(
	securityGroup string,
	rule RuleRequest) (management.OperationID, error) {
	if securityGroup == "" {
		return "", fmt.Errorf(errParamNotSpecified, "securityGroup")
	}
	if rule.Name == "" {
		return "", fmt.Errorf(errParamNotSpecified, "Name")
	}
	if rule.Type == "" {
		return "", fmt.Errorf(errParamNotSpecified, "Type")
	}
	if rule.Priority == 0 {
		return "", fmt.Errorf(errParamNotSpecified, "Priority")
	}
	if rule.Action == "" {
		return "", fmt.Errorf(errParamNotSpecified, "Action")
	}
	if rule.SourceAddressPrefix == "" {
		return "", fmt.Errorf(errParamNotSpecified, "SourceAddressPrefix")
	}
	if rule.SourcePortRange == "" {
		return "", fmt.Errorf(errParamNotSpecified, "SourcePortRange")
	}
	if rule.DestinationAddressPrefix == "" {
		return "", fmt.Errorf(errParamNotSpecified, "DestinationAddressPrefix")
	}
	if rule.DestinationPortRange == "" {
		return "", fmt.Errorf(errParamNotSpecified, "DestinationPortRange")
	}
	if rule.Protocol == "" {
		return "", fmt.Errorf(errParamNotSpecified, "Protocol")
	}

	data, err := xml.Marshal(rule)
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(setSecurityGroupRuleURL, securityGroup, rule.Name)
	return sg.client.SendAzurePutRequest(requestURL, "", data)
}

// DeleteNetworkSecurityGroupRule deletes a network security group rule from
// the specified network security group
//
// https://msdn.microsoft.com/en-us/library/azure/dn913816.aspx
func (sg SecurityGroupClient) DeleteNetworkSecurityGroupRule(
	securityGroup string,
	rule string) (management.OperationID, error) {
	if securityGroup == "" {
		return "", fmt.Errorf(errParamNotSpecified, "securityGroup")
	}
	if rule == "" {
		return "", fmt.Errorf(errParamNotSpecified, "rule")
	}

	requestURL := fmt.Sprintf(deleteSecurityGroupRuleURL, securityGroup, rule)
	return sg.client.SendAzureDeleteRequest(requestURL)
}
