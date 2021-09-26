// +build go1.7

// Package virtualnetwork provides a client for Virtual Networks.
package virtualnetwork

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

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

const (
	azureNetworkConfigurationURL = "services/networking/media"
)

// NewClient is used to return new VirtualNetworkClient instance
func NewClient(client management.Client) VirtualNetworkClient {
	return VirtualNetworkClient{client: client}
}

// GetVirtualNetworkConfiguration retreives the current virtual network
// configuration for the currently active subscription. Note that the
// underlying Azure API means that network related operations are not safe
// for running concurrently.
func (c VirtualNetworkClient) GetVirtualNetworkConfiguration() (NetworkConfiguration, error) {
	networkConfiguration := c.NewNetworkConfiguration()
	response, err := c.client.SendAzureGetRequest(azureNetworkConfigurationURL)
	if err != nil {
		return networkConfiguration, err
	}

	err = xml.Unmarshal(response, &networkConfiguration)
	return networkConfiguration, err

}

// SetVirtualNetworkConfiguration configures the virtual networks for the
// currently active subscription according to the NetworkConfiguration given.
// Note that the underlying Azure API means that network related operations
// are not safe for running concurrently.
func (c VirtualNetworkClient) SetVirtualNetworkConfiguration(networkConfiguration NetworkConfiguration) (management.OperationID, error) {
	networkConfiguration.setXMLNamespaces()
	networkConfigurationBytes, err := xml.Marshal(networkConfiguration)
	if err != nil {
		return "", err
	}

	return c.client.SendAzurePutRequest(azureNetworkConfigurationURL, "text/plain", networkConfigurationBytes)
}
