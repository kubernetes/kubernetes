/*
Copyright (c) 2019 VMware, Inc. All Rights Reserved.

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

	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

// Minimal set of internal types and methods:
// - Fetch() - used by ovftool to collect various managed object properties
// - RetrieveInternalContent() - used by ovftool to obtain a reference to NfcService (which it does not use by default)

func init() {
	types.Add("Fetch", reflect.TypeOf((*Fetch)(nil)).Elem())
}

type Fetch struct {
	This types.ManagedObjectReference `xml:"_this"`
	Prop string                       `xml:"prop"`
}

type FetchResponse struct {
	Returnval types.AnyType `xml:"returnval,omitempty,typeattr"`
}

type FetchBody struct {
	Res    *FetchResponse `xml:"FetchResponse,omitempty"`
	Fault_ *soap.Fault    `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FetchBody) Fault() *soap.Fault { return b.Fault_ }

func init() {
	types.Add("RetrieveInternalContent", reflect.TypeOf((*RetrieveInternalContent)(nil)).Elem())
}

type RetrieveInternalContent struct {
	This types.ManagedObjectReference `xml:"_this"`
}

type RetrieveInternalContentResponse struct {
	Returnval InternalServiceInstanceContent `xml:"returnval"`
}

type RetrieveInternalContentBody struct {
	Res    *RetrieveInternalContentResponse `xml:"RetrieveInternalContentResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *RetrieveInternalContentBody) Fault() *soap.Fault { return b.Fault_ }

type InternalServiceInstanceContent struct {
	types.DynamicData

	NfcService types.ManagedObjectReference `xml:"nfcService"`
}
