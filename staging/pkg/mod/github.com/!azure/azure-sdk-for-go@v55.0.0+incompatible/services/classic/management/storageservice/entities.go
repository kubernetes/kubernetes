// +build go1.7

package storageservice

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"encoding/xml"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

// StorageServiceClient is used to perform operations on Azure Storage
type StorageServiceClient struct {
	client management.Client
}

type ListStorageServicesResponse struct {
	StorageServices []StorageServiceResponse `xml:"StorageService"`
}

type StorageServiceResponse struct {
	URL                      string `xml:"Url"`
	ServiceName              string
	StorageServiceProperties StorageServiceProperties
}

type StorageServiceProperties struct {
	Description           string
	Location              string
	Label                 string
	Status                string
	Endpoints             []string `xml:"Endpoints>Endpoint"`
	GeoReplicationEnabled string
	GeoPrimaryRegion      string
}

type GetStorageServiceKeysResponse struct {
	URL          string `xml:"Url"`
	PrimaryKey   string `xml:"StorageServiceKeys>Primary"`
	SecondaryKey string `xml:"StorageServiceKeys>Secondary"`
}

type CreateStorageServiceInput struct {
	XMLName xml.Name `xml:"http://schemas.microsoft.com/windowsazure CreateStorageServiceInput"`
	StorageAccountCreateParameters
}

type StorageAccountCreateParameters struct {
	ServiceName        string
	Description        string `xml:",omitempty"`
	Label              string
	AffinityGroup      string `xml:",omitempty"`
	Location           string `xml:",omitempty"`
	ExtendedProperties ExtendedPropertyList
	AccountType        AccountType
}

type AccountType string

const (
	AccountTypeStandardLRS   AccountType = "Standard_LRS"
	AccountTypeStandardZRS   AccountType = "Standard_ZRS"
	AccountTypeStandardGRS   AccountType = "Standard_GRS"
	AccountTypeStandardRAGRS AccountType = "Standard_RAGRS"
	AccountTypePremiumLRS    AccountType = "Premium_LRS"
)

type ExtendedPropertyList struct {
	ExtendedProperty []ExtendedProperty
}

type ExtendedProperty struct {
	Name  string
	Value string
}

type AvailabilityResponse struct {
	XMLName xml.Name `xml:"AvailabilityResponse"`
	Xmlns   string   `xml:"xmlns,attr"`
	Result  bool
	Reason  string
}
