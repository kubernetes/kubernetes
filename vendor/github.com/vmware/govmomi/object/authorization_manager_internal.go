/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package object

import (
	"context"

	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type DisabledMethodRequest struct {
	Method string `xml:"method"`
	Reason string `xml:"reasonId"`
}

type disableMethodsRequest struct {
	This   types.ManagedObjectReference   `xml:"_this"`
	Entity []types.ManagedObjectReference `xml:"entity"`
	Method []DisabledMethodRequest        `xml:"method"`
	Source string                         `xml:"sourceId"`
	Scope  bool                           `xml:"sessionScope,omitempty"`
}

type disableMethodsBody struct {
	Req *disableMethodsRequest `xml:"urn:internalvim25 DisableMethods,omitempty"`
	Res interface{}            `xml:"urn:vim25 DisableMethodsResponse,omitempty"`
	Err *soap.Fault            `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *disableMethodsBody) Fault() *soap.Fault { return b.Err }

func (m AuthorizationManager) DisableMethods(ctx context.Context, entity []types.ManagedObjectReference, method []DisabledMethodRequest, source string) error {
	var reqBody, resBody disableMethodsBody

	reqBody.Req = &disableMethodsRequest{
		This:   m.Reference(),
		Entity: entity,
		Method: method,
		Source: source,
	}

	return m.Client().RoundTrip(ctx, &reqBody, &resBody)
}

type enableMethodsRequest struct {
	This   types.ManagedObjectReference   `xml:"_this"`
	Entity []types.ManagedObjectReference `xml:"entity"`
	Method []string                       `xml:"method"`
	Source string                         `xml:"sourceId"`
}

type enableMethodsBody struct {
	Req *enableMethodsRequest `xml:"urn:internalvim25 EnableMethods,omitempty"`
	Res interface{}           `xml:"urn:vim25 EnableMethodsResponse,omitempty"`
	Err *soap.Fault           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *enableMethodsBody) Fault() *soap.Fault { return b.Err }

func (m AuthorizationManager) EnableMethods(ctx context.Context, entity []types.ManagedObjectReference, method []string, source string) error {
	var reqBody, resBody enableMethodsBody

	reqBody.Req = &enableMethodsRequest{
		This:   m.Reference(),
		Entity: entity,
		Method: method,
		Source: source,
	}

	return m.Client().RoundTrip(ctx, &reqBody, &resBody)
}
