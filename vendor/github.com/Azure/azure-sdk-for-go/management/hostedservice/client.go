// +build go1.7

// Package hostedservice provides a client for Hosted Services.
package hostedservice

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
	"encoding/base64"
	"encoding/xml"
	"fmt"

	"github.com/Azure/azure-sdk-for-go/management"
)

const (
	azureXmlns                        = "http://schemas.microsoft.com/windowsazure"
	azureDeploymentListURL            = "services/hostedservices/%s/deployments"
	azureHostedServiceListURL         = "services/hostedservices"
	azureHostedServiceAvailabilityURL = "services/hostedservices/operations/isavailable/%s"
	azureDeploymentURL                = "services/hostedservices/%s/deployments/%s"
	deleteAzureDeploymentURL          = "services/hostedservices/%s/deployments/%s"
	getHostedServicePropertiesURL     = "services/hostedservices/%s"
	azureServiceCertificateURL        = "services/hostedservices/%s/certificates"

	errParamNotSpecified = "Parameter %s is not specified."
)

//NewClient is used to return a handle to the HostedService API
func NewClient(client management.Client) HostedServiceClient {
	return HostedServiceClient{client: client}
}

func (h HostedServiceClient) CreateHostedService(params CreateHostedServiceParameters) error {
	req, err := xml.Marshal(params)
	if err != nil {
		return err
	}

	_, err = h.client.SendAzurePostRequest(azureHostedServiceListURL, req) // not a long running operation
	return err
}

func (h HostedServiceClient) CheckHostedServiceNameAvailability(dnsName string) (AvailabilityResponse, error) {
	var r AvailabilityResponse
	if dnsName == "" {
		return r, fmt.Errorf(errParamNotSpecified, "dnsName")
	}

	requestURL := fmt.Sprintf(azureHostedServiceAvailabilityURL, dnsName)
	response, err := h.client.SendAzureGetRequest(requestURL)
	if err != nil {
		return r, err
	}

	err = xml.Unmarshal(response, &r)
	return r, err
}

func (h HostedServiceClient) DeleteHostedService(dnsName string, deleteDisksAndBlobs bool) (management.OperationID, error) {
	if dnsName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "dnsName")
	}

	requestURL := fmt.Sprintf(getHostedServicePropertiesURL, dnsName)
	if deleteDisksAndBlobs {
		requestURL += "?comp=media"
	}
	return h.client.SendAzureDeleteRequest(requestURL)
}

func (h HostedServiceClient) GetHostedService(name string) (HostedService, error) {
	hostedService := HostedService{}
	if name == "" {
		return hostedService, fmt.Errorf(errParamNotSpecified, "name")
	}

	requestURL := fmt.Sprintf(getHostedServicePropertiesURL, name)
	response, err := h.client.SendAzureGetRequest(requestURL)
	if err != nil {
		return hostedService, err
	}

	err = xml.Unmarshal(response, &hostedService)
	if err != nil {
		return hostedService, err
	}

	decodedLabel, err := base64.StdEncoding.DecodeString(hostedService.LabelBase64)
	if err != nil {
		return hostedService, err
	}
	hostedService.Label = string(decodedLabel)
	return hostedService, nil
}

func (h HostedServiceClient) ListHostedServices() (ListHostedServicesResponse, error) {
	var response ListHostedServicesResponse

	data, err := h.client.SendAzureGetRequest(azureHostedServiceListURL)
	if err != nil {
		return response, err
	}

	err = xml.Unmarshal(data, &response)
	return response, err
}

func (h HostedServiceClient) AddCertificate(dnsName string, certData []byte, certificateFormat CertificateFormat, password string) (management.OperationID, error) {
	if dnsName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "dnsName")
	}

	certBase64 := base64.StdEncoding.EncodeToString(certData)

	addCertificate := CertificateFile{
		Data:              certBase64,
		CertificateFormat: certificateFormat,
		Password:          password,
		Xmlns:             azureXmlns,
	}
	buffer, err := xml.Marshal(addCertificate)
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureServiceCertificateURL, dnsName)
	return h.client.SendAzurePostRequest(requestURL, buffer)
}
