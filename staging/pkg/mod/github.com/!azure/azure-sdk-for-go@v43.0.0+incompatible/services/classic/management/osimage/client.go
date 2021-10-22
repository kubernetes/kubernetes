// +build go1.7

// Package osimage provides a client for Operating System Images.
package osimage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/xml"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

const (
	azureImageListURL    = "services/images"
	errInvalidImage      = "Can not find image %s in specified subscription, please specify another image name."
	errParamNotSpecified = "Parameter %s is not specified."
)

// NewClient is used to instantiate a new OSImageClient from an Azure client.
func NewClient(client management.Client) OSImageClient {
	return OSImageClient{client: client}
}

func (c OSImageClient) ListOSImages() (ListOSImagesResponse, error) {
	var l ListOSImagesResponse

	response, err := c.client.SendAzureGetRequest(azureImageListURL)
	if err != nil {
		return l, err
	}

	err = xml.Unmarshal(response, &l)
	return l, err
}

// AddOSImage adds an operating system image to the image repository that is associated with the specified subscription.
//
// See https://msdn.microsoft.com/en-us/library/azure/jj157192.aspx for details.
func (c OSImageClient) AddOSImage(osi *OSImage) (management.OperationID, error) {
	data, err := xml.Marshal(osi)
	if err != nil {
		return "", err
	}

	return c.client.SendAzurePostRequest(azureImageListURL, data)

}
