// +build go1.7

package virtualmachine

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

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

const (
	azureResourceExtensionsURL     = "services/resourceextensions"
	azureCloudServiceExtensionsURL = "services/hostedservices/%s/extensions"
	azureCloudServiceExtensionURL  = "services/hostedservices/%s/extensions/%s"
)

// GetResourceExtensions lists the resource extensions that are available to add
// to a virtual machine.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn495441.aspx
func (c VirtualMachineClient) GetResourceExtensions() (extensions []ResourceExtension, err error) {
	data, err := c.client.SendAzureGetRequest(azureResourceExtensionsURL)
	if err != nil {
		return extensions, err
	}

	var response ResourceExtensions
	err = xml.Unmarshal(data, &response)
	extensions = response.List
	return
}

// Extensions is a list of extensions returned by the ListExtensions response
type Extensions struct {
	XMLName    xml.Name        `xml:"http://schemas.microsoft.com/windowsazure Extensions"`
	Extensions []ExtensionInfo `xml:"Extension"`
}

// ExtensionInfo defined the type retured by GetExtension
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-get-extension
type ExtensionInfo struct {
	XMLName                     xml.Name `xml:"http://schemas.microsoft.com/windowsazure Extension"`
	ProviderNameSpace           string
	Type                        string
	ID                          string `xml:"Id"`
	Version                     string
	Thumbprint                  string
	PublicConfigurationSchema   string
	ThumbprintAlgorithm         string
	IsJSONExtension             bool `xml:"IsJsonExtension"`
	DisallowMajorVersionUpgrade bool
}

// GetExtension retrieves information about a specified extension that was added to a cloud service.
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-get-extension
func (c VirtualMachineClient) GetExtension(cloudServiceName string, extensionID string) (extension ExtensionInfo, err error) {

	if cloudServiceName == "" {
		return ExtensionInfo{}, fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if extensionID == "" {
		return ExtensionInfo{}, fmt.Errorf(errParamNotSpecified, "extensionID")
	}

	requestURL := fmt.Sprintf(azureCloudServiceExtensionURL, cloudServiceName, extensionID)
	data, err := c.client.SendAzureGetRequest(requestURL)
	if err != nil {
		return ExtensionInfo{}, err
	}
	err = xml.Unmarshal(data, &extension)
	return
}

// ListExtensions lists all of the extensions that were added to a cloud service.
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-list-extensions
func (c VirtualMachineClient) ListExtensions(cloudServiceName string) (extensions []ExtensionInfo, err error) {

	if cloudServiceName == "" {
		return []ExtensionInfo{}, fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}

	requestURL := fmt.Sprintf(azureCloudServiceExtensionsURL, cloudServiceName)
	data, err := c.client.SendAzureGetRequest(requestURL)
	if err != nil {
		return []ExtensionInfo{}, err
	}
	var response Extensions
	err = xml.Unmarshal(data, &response)
	extensions = response.Extensions
	return
}

// AddExtensionOptions defines the options available for adding extensions to a cloud service
type AddExtensionOptions struct {
	ProviderNameSpace    string
	Type                 string
	ID                   string
	Thumbprint           string
	ThumbprintAlgorithm  string
	PublicConfiguration  string
	PrivateConfiguration string
	Version              string
}

// AddExtensionRequest is the type used to submit AddExtension requests
type AddExtensionRequest struct {
	XMLName              xml.Name `xml:"http://schemas.microsoft.com/windowsazure Extension"`
	ProviderNameSpace    string
	Type                 string
	ID                   string `xml:"Id"`
	Thumbprint           string
	ThumbprintAlgorithm  string
	PublicConfiguration  string
	PrivateConfiguration string
	Version              string
}

// AddExtension addes an extension to the cloud service
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-add-extension
func (c VirtualMachineClient) AddExtension(cloudServiceName string, options AddExtensionOptions) (management.OperationID, error) {

	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if options.ID == "" {
		return "", fmt.Errorf(errParamNotSpecified, "options.ID")
	}
	if options.ProviderNameSpace == "" {
		return "", fmt.Errorf(errParamNotSpecified, "options.ProviderNameSpace")
	}
	if options.Type == "" {
		return "", fmt.Errorf(errParamNotSpecified, "options.Type")
	}

	req := AddExtensionRequest{
		ProviderNameSpace:    options.ProviderNameSpace,
		Type:                 options.Type,
		ID:                   options.ID,
		Thumbprint:           options.Thumbprint,
		ThumbprintAlgorithm:  options.ThumbprintAlgorithm,
		PublicConfiguration:  options.PublicConfiguration,
		PrivateConfiguration: options.PrivateConfiguration,
		Version:              options.Version,
	}

	data, err := xml.Marshal(req)
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureCloudServiceExtensionsURL, cloudServiceName)
	return c.client.SendAzurePostRequest(requestURL, data)
}

// DeleteExtension deletes the specified extension from a cloud service.
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-delete-extension
func (c VirtualMachineClient) DeleteExtension(cloudServiceName string, extensionID string) (management.OperationID, error) {

	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if extensionID == "" {
		return "", fmt.Errorf(errParamNotSpecified, "extensionID")
	}

	requestURL := fmt.Sprintf(azureCloudServiceExtensionURL, cloudServiceName, extensionID)
	return c.client.SendAzureDeleteRequest(requestURL)
}
