// +build go1.7

// Package storageservice provides a client for Storage Services.
package storageservice

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
	azureStorageServiceListURL         = "services/storageservices"
	azureStorageServiceURL             = "services/storageservices/%s"
	azureStorageServiceKeysURL         = "services/storageservices/%s/keys"
	azureStorageAccountAvailabilityURL = "services/storageservices/operations/isavailable/%s"

	azureXmlns = "http://schemas.microsoft.com/windowsazure"

	errParamNotSpecified = "Parameter %s is not specified."
)

// NewClient is used to instantiate a new StorageServiceClient from an Azure
// client.
func NewClient(s management.Client) StorageServiceClient {
	return StorageServiceClient{client: s}
}

func (s StorageServiceClient) ListStorageServices() (ListStorageServicesResponse, error) {
	var l ListStorageServicesResponse
	response, err := s.client.SendAzureGetRequest(azureStorageServiceListURL)
	if err != nil {
		return l, err
	}

	err = xml.Unmarshal(response, &l)
	return l, err
}

func (s StorageServiceClient) GetStorageService(serviceName string) (StorageServiceResponse, error) {
	var svc StorageServiceResponse
	if serviceName == "" {
		return svc, fmt.Errorf(errParamNotSpecified, "serviceName")
	}

	requestURL := fmt.Sprintf(azureStorageServiceURL, serviceName)
	response, err := s.client.SendAzureGetRequest(requestURL)
	if err != nil {
		return svc, err
	}

	err = xml.Unmarshal(response, &svc)
	return svc, err
}

func (s StorageServiceClient) GetStorageServiceKeys(serviceName string) (GetStorageServiceKeysResponse, error) {
	var r GetStorageServiceKeysResponse
	if serviceName == "" {
		return r, fmt.Errorf(errParamNotSpecified, "serviceName")
	}

	requestURL := fmt.Sprintf(azureStorageServiceKeysURL, serviceName)
	data, err := s.client.SendAzureGetRequest(requestURL)
	if err != nil {
		return r, err
	}

	err = xml.Unmarshal(data, &r)
	return r, err
}

func (s StorageServiceClient) CreateStorageService(parameters StorageAccountCreateParameters) (management.OperationID, error) {
	data, err := xml.Marshal(CreateStorageServiceInput{
		StorageAccountCreateParameters: parameters})
	if err != nil {
		return "", err
	}

	return s.client.SendAzurePostRequest(azureStorageServiceListURL, data)
}

func (s StorageServiceClient) DeleteStorageService(serviceName string) (management.OperationID, error) {
	if serviceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "serviceName")
	}

	requestURL := fmt.Sprintf(azureStorageServiceURL, serviceName)
	return s.client.SendAzureDeleteRequest(requestURL)
}

// CheckStorageAccountNameAvailability checks to if the specified storage account
// name is available.
//
// See https://msdn.microsoft.com/en-us/library/azure/jj154125.aspx
func (s StorageServiceClient) CheckStorageAccountNameAvailability(name string) (AvailabilityResponse, error) {
	var r AvailabilityResponse
	if name == "" {
		return r, fmt.Errorf(errParamNotSpecified, "name")
	}

	requestURL := fmt.Sprintf(azureStorageAccountAvailabilityURL, name)
	response, err := s.client.SendAzureGetRequest(requestURL)
	if err != nil {
		return r, err
	}

	err = xml.Unmarshal(response, &r)
	return r, err
}
