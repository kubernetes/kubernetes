/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package methods

import (
	"context"

	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type RetrieveDynamicTypeManagerBody struct {
	Req    *types.RetrieveDynamicTypeManager         `xml:"urn:vim25 RetrieveDynamicTypeManager"`
	Res    *types.RetrieveDynamicTypeManagerResponse `xml:"urn:vim25 RetrieveDynamicTypeManagerResponse"`
	Fault_ *soap.Fault
}

func (b *RetrieveDynamicTypeManagerBody) Fault() *soap.Fault { return b.Fault_ }

func RetrieveDynamicTypeManager(ctx context.Context, r soap.RoundTripper, req *types.RetrieveDynamicTypeManager) (*types.RetrieveDynamicTypeManagerResponse, error) {
	var reqBody, resBody RetrieveDynamicTypeManagerBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type RetrieveManagedMethodExecuterBody struct {
	Req    *types.RetrieveManagedMethodExecuter         `xml:"urn:vim25 RetrieveManagedMethodExecuter"`
	Res    *types.RetrieveManagedMethodExecuterResponse `xml:"urn:vim25 RetrieveManagedMethodExecuterResponse"`
	Fault_ *soap.Fault
}

func (b *RetrieveManagedMethodExecuterBody) Fault() *soap.Fault { return b.Fault_ }

func RetrieveManagedMethodExecuter(ctx context.Context, r soap.RoundTripper, req *types.RetrieveManagedMethodExecuter) (*types.RetrieveManagedMethodExecuterResponse, error) {
	var reqBody, resBody RetrieveManagedMethodExecuterBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type DynamicTypeMgrQueryMoInstancesBody struct {
	Req    *types.DynamicTypeMgrQueryMoInstances         `xml:"urn:vim25 DynamicTypeMgrQueryMoInstances"`
	Res    *types.DynamicTypeMgrQueryMoInstancesResponse `xml:"urn:vim25 DynamicTypeMgrQueryMoInstancesResponse"`
	Fault_ *soap.Fault
}

func (b *DynamicTypeMgrQueryMoInstancesBody) Fault() *soap.Fault { return b.Fault_ }

func DynamicTypeMgrQueryMoInstances(ctx context.Context, r soap.RoundTripper, req *types.DynamicTypeMgrQueryMoInstances) (*types.DynamicTypeMgrQueryMoInstancesResponse, error) {
	var reqBody, resBody DynamicTypeMgrQueryMoInstancesBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type DynamicTypeMgrQueryTypeInfoBody struct {
	Req    *types.DynamicTypeMgrQueryTypeInfo         `xml:"urn:vim25 DynamicTypeMgrQueryTypeInfo"`
	Res    *types.DynamicTypeMgrQueryTypeInfoResponse `xml:"urn:vim25 DynamicTypeMgrQueryTypeInfoResponse"`
	Fault_ *soap.Fault
}

func (b *DynamicTypeMgrQueryTypeInfoBody) Fault() *soap.Fault { return b.Fault_ }

func DynamicTypeMgrQueryTypeInfo(ctx context.Context, r soap.RoundTripper, req *types.DynamicTypeMgrQueryTypeInfo) (*types.DynamicTypeMgrQueryTypeInfoResponse, error) {
	var reqBody, resBody DynamicTypeMgrQueryTypeInfoBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type ExecuteSoapBody struct {
	Req    *types.ExecuteSoap         `xml:"urn:vim25 ExecuteSoap"`
	Res    *types.ExecuteSoapResponse `xml:"urn:vim25 ExecuteSoapResponse"`
	Fault_ *soap.Fault
}

func (b *ExecuteSoapBody) Fault() *soap.Fault { return b.Fault_ }

func ExecuteSoap(ctx context.Context, r soap.RoundTripper, req *types.ExecuteSoap) (*types.ExecuteSoapResponse, error) {
	var reqBody, resBody ExecuteSoapBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}
