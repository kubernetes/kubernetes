// +build go1.7

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

	"github.com/Azure/azure-sdk-for-go/management"
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
