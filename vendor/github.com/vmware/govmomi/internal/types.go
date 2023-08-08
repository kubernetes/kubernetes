/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package internal

import (
	"reflect"

	"github.com/vmware/govmomi/vim25/types"
)

type DynamicTypeMgrQueryMoInstancesRequest struct {
	This       types.ManagedObjectReference `xml:"_this"`
	FilterSpec BaseDynamicTypeMgrFilterSpec `xml:"filterSpec,omitempty,typeattr"`
}

type DynamicTypeMgrQueryMoInstancesResponse struct {
	Returnval []DynamicTypeMgrMoInstance `xml:"urn:vim25 returnval"`
}

type DynamicTypeEnumTypeInfo struct {
	types.DynamicData

	Name       string                     `xml:"name"`
	WsdlName   string                     `xml:"wsdlName"`
	Version    string                     `xml:"version"`
	Value      []string                   `xml:"value,omitempty"`
	Annotation []DynamicTypeMgrAnnotation `xml:"annotation,omitempty"`
}

func init() {
	types.Add("DynamicTypeEnumTypeInfo", reflect.TypeOf((*DynamicTypeEnumTypeInfo)(nil)).Elem())
}

type DynamicTypeMgrAllTypeInfoRequest struct {
	types.DynamicData

	ManagedTypeInfo []DynamicTypeMgrManagedTypeInfo `xml:"managedTypeInfo,omitempty"`
	EnumTypeInfo    []DynamicTypeEnumTypeInfo       `xml:"enumTypeInfo,omitempty"`
	DataTypeInfo    []DynamicTypeMgrDataTypeInfo    `xml:"dataTypeInfo,omitempty"`
}

func init() {
	types.Add("DynamicTypeMgrAllTypeInfo", reflect.TypeOf((*DynamicTypeMgrAllTypeInfoRequest)(nil)).Elem())
}

type DynamicTypeMgrAnnotation struct {
	types.DynamicData

	Name      string   `xml:"name"`
	Parameter []string `xml:"parameter,omitempty"`
}

func init() {
	types.Add("DynamicTypeMgrAnnotation", reflect.TypeOf((*DynamicTypeMgrAnnotation)(nil)).Elem())
}

type DynamicTypeMgrDataTypeInfo struct {
	types.DynamicData

	Name       string                           `xml:"name"`
	WsdlName   string                           `xml:"wsdlName"`
	Version    string                           `xml:"version"`
	Base       []string                         `xml:"base,omitempty"`
	Property   []DynamicTypeMgrPropertyTypeInfo `xml:"property,omitempty"`
	Annotation []DynamicTypeMgrAnnotation       `xml:"annotation,omitempty"`
}

func init() {
	types.Add("DynamicTypeMgrDataTypeInfo", reflect.TypeOf((*DynamicTypeMgrDataTypeInfo)(nil)).Elem())
}

func (b *DynamicTypeMgrFilterSpec) GetDynamicTypeMgrFilterSpec() *DynamicTypeMgrFilterSpec { return b }

type BaseDynamicTypeMgrFilterSpec interface {
	GetDynamicTypeMgrFilterSpec() *DynamicTypeMgrFilterSpec
}

type DynamicTypeMgrFilterSpec struct {
	types.DynamicData
}

func init() {
	types.Add("DynamicTypeMgrFilterSpec", reflect.TypeOf((*DynamicTypeMgrFilterSpec)(nil)).Elem())
}

type DynamicTypeMgrManagedTypeInfo struct {
	types.DynamicData

	Name       string                           `xml:"name"`
	WsdlName   string                           `xml:"wsdlName"`
	Version    string                           `xml:"version"`
	Base       []string                         `xml:"base,omitempty"`
	Property   []DynamicTypeMgrPropertyTypeInfo `xml:"property,omitempty"`
	Method     []DynamicTypeMgrMethodTypeInfo   `xml:"method,omitempty"`
	Annotation []DynamicTypeMgrAnnotation       `xml:"annotation,omitempty"`
}

func init() {
	types.Add("DynamicTypeMgrManagedTypeInfo", reflect.TypeOf((*DynamicTypeMgrManagedTypeInfo)(nil)).Elem())
}

type DynamicTypeMgrMethodTypeInfo struct {
	types.DynamicData

	Name           string                        `xml:"name"`
	WsdlName       string                        `xml:"wsdlName"`
	Version        string                        `xml:"version"`
	ParamTypeInfo  []DynamicTypeMgrParamTypeInfo `xml:"paramTypeInfo,omitempty"`
	ReturnTypeInfo *DynamicTypeMgrParamTypeInfo  `xml:"returnTypeInfo,omitempty"`
	Fault          []string                      `xml:"fault,omitempty"`
	PrivId         string                        `xml:"privId,omitempty"`
	Annotation     []DynamicTypeMgrAnnotation    `xml:"annotation,omitempty"`
}

func init() {
	types.Add("DynamicTypeMgrMethodTypeInfo", reflect.TypeOf((*DynamicTypeMgrMethodTypeInfo)(nil)).Elem())
}

type DynamicTypeMgrMoFilterSpec struct {
	DynamicTypeMgrFilterSpec

	Id         string `xml:"id,omitempty"`
	TypeSubstr string `xml:"typeSubstr,omitempty"`
}

func init() {
	types.Add("DynamicTypeMgrMoFilterSpec", reflect.TypeOf((*DynamicTypeMgrMoFilterSpec)(nil)).Elem())
}

type DynamicTypeMgrMoInstance struct {
	types.DynamicData

	Id     string `xml:"id"`
	MoType string `xml:"moType"`
}

func init() {
	types.Add("DynamicTypeMgrMoInstance", reflect.TypeOf((*DynamicTypeMgrMoInstance)(nil)).Elem())
}

type DynamicTypeMgrParamTypeInfo struct {
	types.DynamicData

	Name       string                     `xml:"name"`
	Version    string                     `xml:"version"`
	Type       string                     `xml:"type"`
	PrivId     string                     `xml:"privId,omitempty"`
	Annotation []DynamicTypeMgrAnnotation `xml:"annotation,omitempty"`
}

func init() {
	types.Add("DynamicTypeMgrParamTypeInfo", reflect.TypeOf((*DynamicTypeMgrParamTypeInfo)(nil)).Elem())
}

type DynamicTypeMgrPropertyTypeInfo struct {
	types.DynamicData

	Name        string                     `xml:"name"`
	Version     string                     `xml:"version"`
	Type        string                     `xml:"type"`
	PrivId      string                     `xml:"privId,omitempty"`
	MsgIdFormat string                     `xml:"msgIdFormat,omitempty"`
	Annotation  []DynamicTypeMgrAnnotation `xml:"annotation,omitempty"`
}

type DynamicTypeMgrQueryTypeInfoRequest struct {
	This       types.ManagedObjectReference `xml:"_this"`
	FilterSpec BaseDynamicTypeMgrFilterSpec `xml:"filterSpec,omitempty,typeattr"`
}

type DynamicTypeMgrQueryTypeInfoResponse struct {
	Returnval DynamicTypeMgrAllTypeInfoRequest `xml:"urn:vim25 returnval"`
}

func init() {
	types.Add("DynamicTypeMgrPropertyTypeInfo", reflect.TypeOf((*DynamicTypeMgrPropertyTypeInfo)(nil)).Elem())
}

type DynamicTypeMgrTypeFilterSpec struct {
	DynamicTypeMgrFilterSpec

	TypeSubstr string `xml:"typeSubstr,omitempty"`
}

func init() {
	types.Add("DynamicTypeMgrTypeFilterSpec", reflect.TypeOf((*DynamicTypeMgrTypeFilterSpec)(nil)).Elem())
}

type ReflectManagedMethodExecuterSoapArgument struct {
	types.DynamicData

	Name string `xml:"name"`
	Val  string `xml:"val"`
}

func init() {
	types.Add("ReflectManagedMethodExecuterSoapArgument", reflect.TypeOf((*ReflectManagedMethodExecuterSoapArgument)(nil)).Elem())
}

type ReflectManagedMethodExecuterSoapFault struct {
	types.DynamicData

	FaultMsg    string `xml:"faultMsg"`
	FaultDetail string `xml:"faultDetail,omitempty"`
}

func init() {
	types.Add("ReflectManagedMethodExecuterSoapFault", reflect.TypeOf((*ReflectManagedMethodExecuterSoapFault)(nil)).Elem())
}

type ReflectManagedMethodExecuterSoapResult struct {
	types.DynamicData

	Response string                                 `xml:"response,omitempty"`
	Fault    *ReflectManagedMethodExecuterSoapFault `xml:"fault,omitempty"`
}

type RetrieveDynamicTypeManagerRequest struct {
	This types.ManagedObjectReference `xml:"_this"`
}

type RetrieveDynamicTypeManagerResponse struct {
	Returnval *InternalDynamicTypeManager `xml:"urn:vim25 returnval"`
}

type RetrieveManagedMethodExecuterRequest struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("RetrieveManagedMethodExecuter", reflect.TypeOf((*RetrieveManagedMethodExecuterRequest)(nil)).Elem())
}

type RetrieveManagedMethodExecuterResponse struct {
	Returnval *ReflectManagedMethodExecuter `xml:"urn:vim25 returnval"`
}

type InternalDynamicTypeManager struct {
	types.ManagedObjectReference
}

type ReflectManagedMethodExecuter struct {
	types.ManagedObjectReference
}

type ExecuteSoapRequest struct {
	This     types.ManagedObjectReference               `xml:"_this"`
	Moid     string                                     `xml:"moid"`
	Version  string                                     `xml:"version"`
	Method   string                                     `xml:"method"`
	Argument []ReflectManagedMethodExecuterSoapArgument `xml:"argument,omitempty"`
}

type ExecuteSoapResponse struct {
	Returnval *ReflectManagedMethodExecuterSoapResult `xml:"urn:vim25 returnval"`
}
