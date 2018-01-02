// +build go1.7

// Package virtualmachineimage provides a client for Virtual Machine Images.
package virtualmachineimage

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
	"net/url"

	"github.com/Azure/azure-sdk-for-go/management"
)

const (
	azureImageListURL         = "services/vmimages"
	azureImageDeleteURLformat = "services/vmimages/%s"
	azureRoleOperationsURL    = "services/hostedservices/%s/deployments/%s/roleinstances/%s/operations"
	errParamNotSpecified      = "Parameter %s is not specified."
)

//NewClient is used to instantiate a new Client from an Azure client
func NewClient(client management.Client) Client {
	return Client{client}
}

//ListVirtualMachineImages lists the available VM images, filtered by the optional parameters.
//See https://msdn.microsoft.com/en-us/library/azure/dn499770.aspx
func (c Client) ListVirtualMachineImages(parameters ListParameters) (ListVirtualMachineImagesResponse, error) {
	var imageList ListVirtualMachineImagesResponse

	listURL := azureImageListURL

	v := url.Values{}
	if parameters.Location != "" {
		v.Add("location", parameters.Location)
	}

	if parameters.Publisher != "" {
		v.Add("publisher", parameters.Publisher)
	}

	if parameters.Category != "" {
		v.Add("category", parameters.Category)
	}

	query := v.Encode()
	if query != "" {
		listURL = listURL + "?" + query
	}

	response, err := c.SendAzureGetRequest(listURL)
	if err != nil {
		return imageList, err
	}
	err = xml.Unmarshal(response, &imageList)
	return imageList, err
}

//DeleteVirtualMachineImage deletes the named VM image. If deleteVHDs is specified,
//the referenced OS and data disks are also deleted.
//See https://msdn.microsoft.com/en-us/library/azure/dn499769.aspx
func (c Client) DeleteVirtualMachineImage(name string, deleteVHDs bool) error {
	if name == "" {
		return fmt.Errorf(errParamNotSpecified, "name")
	}

	uri := fmt.Sprintf(azureImageDeleteURLformat, name)

	if deleteVHDs {
		uri = uri + "?comp=media"
	}

	_, err := c.SendAzureDeleteRequest(uri) // delete is synchronous for this operation
	return err
}

type ListParameters struct {
	Location  string
	Publisher string
	Category  string
}

const CategoryUser = "User"

//Capture captures a VM into a VM image. The VM has to be shut down previously.
//See https://msdn.microsoft.com/en-us/library/azure/dn499768.aspx
func (c Client) Capture(cloudServiceName, deploymentName, roleName string,
	name, label string, osState OSState, parameters CaptureParameters) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentName")
	}
	if roleName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "roleName")
	}

	request := CaptureRoleAsVMImageOperation{
		VMImageName:       name,
		VMImageLabel:      label,
		OSState:           osState,
		CaptureParameters: parameters,
	}
	data, err := xml.Marshal(request)
	if err != nil {
		return "", err
	}

	return c.SendAzurePostRequest(fmt.Sprintf(azureRoleOperationsURL,
		cloudServiceName, deploymentName, roleName), data)
}
