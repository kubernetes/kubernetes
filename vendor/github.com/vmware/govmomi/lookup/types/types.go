/*
Copyright (c) 2014-2018 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package types

import (
	"reflect"

	"github.com/vmware/govmomi/vim25/types"
	vim "github.com/vmware/govmomi/vim25/types"
)

type Create CreateRequestType

func init() {
	types.Add("lookup:Create", reflect.TypeOf((*Create)(nil)).Elem())
}

type CreateRequestType struct {
	This       vim.ManagedObjectReference          `xml:"_this"`
	ServiceId  string                              `xml:"serviceId"`
	CreateSpec LookupServiceRegistrationCreateSpec `xml:"createSpec"`
}

func init() {
	types.Add("lookup:CreateRequestType", reflect.TypeOf((*CreateRequestType)(nil)).Elem())
}

type CreateResponse struct {
}

type Delete DeleteRequestType

func init() {
	types.Add("lookup:Delete", reflect.TypeOf((*Delete)(nil)).Elem())
}

type DeleteRequestType struct {
	This      vim.ManagedObjectReference `xml:"_this"`
	ServiceId string                     `xml:"serviceId"`
}

func init() {
	types.Add("lookup:DeleteRequestType", reflect.TypeOf((*DeleteRequestType)(nil)).Elem())
}

type DeleteResponse struct {
}

type Get GetRequestType

func init() {
	types.Add("lookup:Get", reflect.TypeOf((*Get)(nil)).Elem())
}

type GetLocale GetLocaleRequestType

func init() {
	types.Add("lookup:GetLocale", reflect.TypeOf((*GetLocale)(nil)).Elem())
}

type GetLocaleRequestType struct {
	This vim.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("lookup:GetLocaleRequestType", reflect.TypeOf((*GetLocaleRequestType)(nil)).Elem())
}

type GetLocaleResponse struct {
	Returnval string `xml:"returnval"`
}

type GetRequestType struct {
	This      vim.ManagedObjectReference `xml:"_this"`
	ServiceId string                     `xml:"serviceId"`
}

func init() {
	types.Add("lookup:GetRequestType", reflect.TypeOf((*GetRequestType)(nil)).Elem())
}

type GetResponse struct {
	Returnval LookupServiceRegistrationInfo `xml:"returnval"`
}

type GetSiteId GetSiteIdRequestType

func init() {
	types.Add("lookup:GetSiteId", reflect.TypeOf((*GetSiteId)(nil)).Elem())
}

type GetSiteIdRequestType struct {
	This vim.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("lookup:GetSiteIdRequestType", reflect.TypeOf((*GetSiteIdRequestType)(nil)).Elem())
}

type GetSiteIdResponse struct {
	Returnval string `xml:"returnval"`
}

type List ListRequestType

func init() {
	types.Add("lookup:List", reflect.TypeOf((*List)(nil)).Elem())
}

type ListRequestType struct {
	This           vim.ManagedObjectReference       `xml:"_this"`
	FilterCriteria *LookupServiceRegistrationFilter `xml:"filterCriteria,omitempty"`
}

func init() {
	types.Add("lookup:ListRequestType", reflect.TypeOf((*ListRequestType)(nil)).Elem())
}

type ListResponse struct {
	Returnval []LookupServiceRegistrationInfo `xml:"returnval,omitempty"`
}

type LookupFaultEntryExistsFault struct {
	LookupFaultServiceFault

	Name string `xml:"name"`
}

func init() {
	types.Add("lookup:LookupFaultEntryExistsFault", reflect.TypeOf((*LookupFaultEntryExistsFault)(nil)).Elem())
}

type LookupFaultEntryExistsFaultFault LookupFaultEntryExistsFault

func init() {
	types.Add("lookup:LookupFaultEntryExistsFaultFault", reflect.TypeOf((*LookupFaultEntryExistsFaultFault)(nil)).Elem())
}

type LookupFaultEntryNotFoundFault struct {
	LookupFaultServiceFault

	Name string `xml:"name"`
}

func init() {
	types.Add("lookup:LookupFaultEntryNotFoundFault", reflect.TypeOf((*LookupFaultEntryNotFoundFault)(nil)).Elem())
}

type LookupFaultEntryNotFoundFaultFault LookupFaultEntryNotFoundFault

func init() {
	types.Add("lookup:LookupFaultEntryNotFoundFaultFault", reflect.TypeOf((*LookupFaultEntryNotFoundFaultFault)(nil)).Elem())
}

type LookupFaultServiceFault struct {
	vim.MethodFault

	ErrorMessage string `xml:"errorMessage,omitempty"`
}

func init() {
	types.Add("lookup:LookupFaultServiceFault", reflect.TypeOf((*LookupFaultServiceFault)(nil)).Elem())
}

type LookupFaultUnsupportedSiteFault struct {
	LookupFaultServiceFault

	OperatingSite string `xml:"operatingSite"`
	RequestedSite string `xml:"requestedSite"`
}

func init() {
	types.Add("lookup:LookupFaultUnsupportedSiteFault", reflect.TypeOf((*LookupFaultUnsupportedSiteFault)(nil)).Elem())
}

type LookupFaultUnsupportedSiteFaultFault LookupFaultUnsupportedSiteFault

func init() {
	types.Add("lookup:LookupFaultUnsupportedSiteFaultFault", reflect.TypeOf((*LookupFaultUnsupportedSiteFaultFault)(nil)).Elem())
}

type LookupHaBackupNodeConfiguration struct {
	vim.DynamicData

	DbType    string `xml:"dbType"`
	DbJdbcUrl string `xml:"dbJdbcUrl"`
	DbUser    string `xml:"dbUser"`
	DbPass    string `xml:"dbPass"`
}

func init() {
	types.Add("lookup:LookupHaBackupNodeConfiguration", reflect.TypeOf((*LookupHaBackupNodeConfiguration)(nil)).Elem())
}

type LookupServiceContent struct {
	vim.DynamicData

	LookupService                vim.ManagedObjectReference  `xml:"lookupService"`
	ServiceRegistration          *vim.ManagedObjectReference `xml:"serviceRegistration,omitempty"`
	DeploymentInformationService vim.ManagedObjectReference  `xml:"deploymentInformationService"`
	L10n                         vim.ManagedObjectReference  `xml:"l10n"`
}

func init() {
	types.Add("lookup:LookupServiceContent", reflect.TypeOf((*LookupServiceContent)(nil)).Elem())
}

type LookupServiceRegistrationAttribute struct {
	vim.DynamicData

	Key   string `xml:"key"`
	Value string `xml:"value"`
}

func init() {
	types.Add("lookup:LookupServiceRegistrationAttribute", reflect.TypeOf((*LookupServiceRegistrationAttribute)(nil)).Elem())
}

type LookupServiceRegistrationCommonServiceInfo struct {
	LookupServiceRegistrationMutableServiceInfo

	OwnerId     string                               `xml:"ownerId"`
	ServiceType LookupServiceRegistrationServiceType `xml:"serviceType"`
	NodeId      string                               `xml:"nodeId,omitempty"`
}

func init() {
	types.Add("lookup:LookupServiceRegistrationCommonServiceInfo", reflect.TypeOf((*LookupServiceRegistrationCommonServiceInfo)(nil)).Elem())
}

type LookupServiceRegistrationCreateSpec struct {
	LookupServiceRegistrationCommonServiceInfo
}

func init() {
	types.Add("lookup:LookupServiceRegistrationCreateSpec", reflect.TypeOf((*LookupServiceRegistrationCreateSpec)(nil)).Elem())
}

type LookupServiceRegistrationEndpoint struct {
	vim.DynamicData

	Url                string                                `xml:"url"`
	EndpointType       LookupServiceRegistrationEndpointType `xml:"endpointType"`
	SslTrust           []string                              `xml:"sslTrust,omitempty"`
	EndpointAttributes []LookupServiceRegistrationAttribute  `xml:"endpointAttributes,omitempty"`
}

func init() {
	types.Add("lookup:LookupServiceRegistrationEndpoint", reflect.TypeOf((*LookupServiceRegistrationEndpoint)(nil)).Elem())
}

type LookupServiceRegistrationEndpointType struct {
	vim.DynamicData

	Protocol string `xml:"protocol,omitempty"`
	Type     string `xml:"type,omitempty"`
}

func init() {
	types.Add("lookup:LookupServiceRegistrationEndpointType", reflect.TypeOf((*LookupServiceRegistrationEndpointType)(nil)).Elem())
}

type LookupServiceRegistrationFilter struct {
	vim.DynamicData

	SiteId       string                                 `xml:"siteId,omitempty"`
	NodeId       string                                 `xml:"nodeId,omitempty"`
	ServiceType  *LookupServiceRegistrationServiceType  `xml:"serviceType,omitempty"`
	EndpointType *LookupServiceRegistrationEndpointType `xml:"endpointType,omitempty"`
}

func init() {
	types.Add("lookup:LookupServiceRegistrationFilter", reflect.TypeOf((*LookupServiceRegistrationFilter)(nil)).Elem())
}

type LookupServiceRegistrationInfo struct {
	LookupServiceRegistrationCommonServiceInfo

	ServiceId string `xml:"serviceId"`
	SiteId    string `xml:"siteId"`
}

func init() {
	types.Add("lookup:LookupServiceRegistrationInfo", reflect.TypeOf((*LookupServiceRegistrationInfo)(nil)).Elem())
}

type LookupServiceRegistrationMutableServiceInfo struct {
	vim.DynamicData

	ServiceVersion                string                               `xml:"serviceVersion"`
	VendorNameResourceKey         string                               `xml:"vendorNameResourceKey,omitempty"`
	VendorNameDefault             string                               `xml:"vendorNameDefault,omitempty"`
	VendorProductInfoResourceKey  string                               `xml:"vendorProductInfoResourceKey,omitempty"`
	VendorProductInfoDefault      string                               `xml:"vendorProductInfoDefault,omitempty"`
	ServiceEndpoints              []LookupServiceRegistrationEndpoint  `xml:"serviceEndpoints,omitempty"`
	ServiceAttributes             []LookupServiceRegistrationAttribute `xml:"serviceAttributes,omitempty"`
	ServiceNameResourceKey        string                               `xml:"serviceNameResourceKey,omitempty"`
	ServiceNameDefault            string                               `xml:"serviceNameDefault,omitempty"`
	ServiceDescriptionResourceKey string                               `xml:"serviceDescriptionResourceKey,omitempty"`
	ServiceDescriptionDefault     string                               `xml:"serviceDescriptionDefault,omitempty"`
}

func init() {
	types.Add("lookup:LookupServiceRegistrationMutableServiceInfo", reflect.TypeOf((*LookupServiceRegistrationMutableServiceInfo)(nil)).Elem())
}

type LookupServiceRegistrationServiceType struct {
	vim.DynamicData

	Product string `xml:"product"`
	Type    string `xml:"type"`
}

func init() {
	types.Add("lookup:LookupServiceRegistrationServiceType", reflect.TypeOf((*LookupServiceRegistrationServiceType)(nil)).Elem())
}

type LookupServiceRegistrationSetSpec struct {
	LookupServiceRegistrationMutableServiceInfo
}

func init() {
	types.Add("lookup:LookupServiceRegistrationSetSpec", reflect.TypeOf((*LookupServiceRegistrationSetSpec)(nil)).Elem())
}

type RetrieveHaBackupConfiguration RetrieveHaBackupConfigurationRequestType

func init() {
	types.Add("lookup:RetrieveHaBackupConfiguration", reflect.TypeOf((*RetrieveHaBackupConfiguration)(nil)).Elem())
}

type RetrieveHaBackupConfigurationRequestType struct {
	This vim.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("lookup:RetrieveHaBackupConfigurationRequestType", reflect.TypeOf((*RetrieveHaBackupConfigurationRequestType)(nil)).Elem())
}

type RetrieveHaBackupConfigurationResponse struct {
	Returnval LookupHaBackupNodeConfiguration `xml:"returnval"`
}

type RetrieveServiceContent RetrieveServiceContentRequestType

func init() {
	types.Add("lookup:RetrieveServiceContent", reflect.TypeOf((*RetrieveServiceContent)(nil)).Elem())
}

type RetrieveServiceContentRequestType struct {
	This vim.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("lookup:RetrieveServiceContentRequestType", reflect.TypeOf((*RetrieveServiceContentRequestType)(nil)).Elem())
}

type RetrieveServiceContentResponse struct {
	Returnval LookupServiceContent `xml:"returnval"`
}

type Set SetRequestType

func init() {
	types.Add("lookup:Set", reflect.TypeOf((*Set)(nil)).Elem())
}

type SetLocale SetLocaleRequestType

func init() {
	types.Add("lookup:SetLocale", reflect.TypeOf((*SetLocale)(nil)).Elem())
}

type SetLocaleRequestType struct {
	This   vim.ManagedObjectReference `xml:"_this"`
	Locale string                     `xml:"locale"`
}

func init() {
	types.Add("lookup:SetLocaleRequestType", reflect.TypeOf((*SetLocaleRequestType)(nil)).Elem())
}

type SetLocaleResponse struct {
	Returnval string `xml:"returnval"`
}

type SetRequestType struct {
	This        vim.ManagedObjectReference       `xml:"_this"`
	ServiceId   string                           `xml:"serviceId"`
	ServiceSpec LookupServiceRegistrationSetSpec `xml:"serviceSpec"`
}

func init() {
	types.Add("lookup:SetRequestType", reflect.TypeOf((*SetRequestType)(nil)).Elem())
}

type SetResponse struct {
}
