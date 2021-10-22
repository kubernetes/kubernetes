// +build go1.7

package hostedservice

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

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
