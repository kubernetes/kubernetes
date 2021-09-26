// +build go1.7

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
	"encoding/xml"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

//HostedServiceClient is used to perform operations on Azure Hosted Services
type HostedServiceClient struct {
	client management.Client
}

type CreateHostedServiceParameters struct {
	XMLName        xml.Name `xml:"http://schemas.microsoft.com/windowsazure CreateHostedService"`
	ServiceName    string
	Label          string
	Description    string
	Location       string
	ReverseDNSFqdn string `xml:"ReverseDnsFqdn,omitempty"`
}

type AvailabilityResponse struct {
	Xmlns  string `xml:"xmlns,attr"`
	Result bool
	Reason string
}

type HostedService struct {
	URL                               string `xml:"Url"`
	ServiceName                       string
	Description                       string `xml:"HostedServiceProperties>Description"`
	AffinityGroup                     string `xml:"HostedServiceProperties>AffinityGroup"`
	Location                          string `xml:"HostedServiceProperties>Location"`
	LabelBase64                       string `xml:"HostedServiceProperties>Label"`
	Label                             string
	Status                            string `xml:"HostedServiceProperties>Status"`
	ReverseDNSFqdn                    string `xml:"HostedServiceProperties>ReverseDnsFqdn"`
	DefaultWinRmCertificateThumbprint string
}

type CertificateFile struct {
	Xmlns             string `xml:"xmlns,attr"`
	Data              string
	CertificateFormat CertificateFormat
	Password          string `xml:",omitempty"`
}

type CertificateFormat string

const (
	CertificateFormatPfx = CertificateFormat("pfx")
	CertificateFormatCer = CertificateFormat("cer")
)

type ListHostedServicesResponse struct {
	HostedServices []HostedService `xml:"HostedService"`
}
