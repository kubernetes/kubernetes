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

package methods

import (
	"context"

	"github.com/vmware/govmomi/lookup/types"
	"github.com/vmware/govmomi/vim25/soap"
)

type CreateBody struct {
	Req    *types.Create         `xml:"urn:lookup Create,omitempty"`
	Res    *types.CreateResponse `xml:"urn:lookup CreateResponse,omitempty"`
	Fault_ *soap.Fault           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *CreateBody) Fault() *soap.Fault { return b.Fault_ }

func Create(ctx context.Context, r soap.RoundTripper, req *types.Create) (*types.CreateResponse, error) {
	var reqBody, resBody CreateBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type DeleteBody struct {
	Req    *types.Delete         `xml:"urn:lookup Delete,omitempty"`
	Res    *types.DeleteResponse `xml:"urn:lookup DeleteResponse,omitempty"`
	Fault_ *soap.Fault           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *DeleteBody) Fault() *soap.Fault { return b.Fault_ }

func Delete(ctx context.Context, r soap.RoundTripper, req *types.Delete) (*types.DeleteResponse, error) {
	var reqBody, resBody DeleteBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetBody struct {
	Req    *types.Get         `xml:"urn:lookup Get,omitempty"`
	Res    *types.GetResponse `xml:"urn:lookup GetResponse,omitempty"`
	Fault_ *soap.Fault        `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetBody) Fault() *soap.Fault { return b.Fault_ }

func Get(ctx context.Context, r soap.RoundTripper, req *types.Get) (*types.GetResponse, error) {
	var reqBody, resBody GetBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetLocaleBody struct {
	Req    *types.GetLocale         `xml:"urn:lookup GetLocale,omitempty"`
	Res    *types.GetLocaleResponse `xml:"urn:lookup GetLocaleResponse,omitempty"`
	Fault_ *soap.Fault              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetLocaleBody) Fault() *soap.Fault { return b.Fault_ }

func GetLocale(ctx context.Context, r soap.RoundTripper, req *types.GetLocale) (*types.GetLocaleResponse, error) {
	var reqBody, resBody GetLocaleBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetSiteIdBody struct {
	Req    *types.GetSiteId         `xml:"urn:lookup GetSiteId,omitempty"`
	Res    *types.GetSiteIdResponse `xml:"urn:lookup GetSiteIdResponse,omitempty"`
	Fault_ *soap.Fault              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetSiteIdBody) Fault() *soap.Fault { return b.Fault_ }

func GetSiteId(ctx context.Context, r soap.RoundTripper, req *types.GetSiteId) (*types.GetSiteIdResponse, error) {
	var reqBody, resBody GetSiteIdBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type ListBody struct {
	Req    *types.List         `xml:"urn:lookup List,omitempty"`
	Res    *types.ListResponse `xml:"urn:lookup ListResponse,omitempty"`
	Fault_ *soap.Fault         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *ListBody) Fault() *soap.Fault { return b.Fault_ }

func List(ctx context.Context, r soap.RoundTripper, req *types.List) (*types.ListResponse, error) {
	var reqBody, resBody ListBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type RetrieveHaBackupConfigurationBody struct {
	Req    *types.RetrieveHaBackupConfiguration         `xml:"urn:lookup RetrieveHaBackupConfiguration,omitempty"`
	Res    *types.RetrieveHaBackupConfigurationResponse `xml:"urn:lookup RetrieveHaBackupConfigurationResponse,omitempty"`
	Fault_ *soap.Fault                                  `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *RetrieveHaBackupConfigurationBody) Fault() *soap.Fault { return b.Fault_ }

func RetrieveHaBackupConfiguration(ctx context.Context, r soap.RoundTripper, req *types.RetrieveHaBackupConfiguration) (*types.RetrieveHaBackupConfigurationResponse, error) {
	var reqBody, resBody RetrieveHaBackupConfigurationBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type RetrieveServiceContentBody struct {
	Req    *types.RetrieveServiceContent         `xml:"urn:lookup RetrieveServiceContent,omitempty"`
	Res    *types.RetrieveServiceContentResponse `xml:"urn:lookup RetrieveServiceContentResponse,omitempty"`
	Fault_ *soap.Fault                           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *RetrieveServiceContentBody) Fault() *soap.Fault { return b.Fault_ }

func RetrieveServiceContent(ctx context.Context, r soap.RoundTripper, req *types.RetrieveServiceContent) (*types.RetrieveServiceContentResponse, error) {
	var reqBody, resBody RetrieveServiceContentBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetBody struct {
	Req    *types.Set         `xml:"urn:lookup Set,omitempty"`
	Res    *types.SetResponse `xml:"urn:lookup SetResponse,omitempty"`
	Fault_ *soap.Fault        `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetBody) Fault() *soap.Fault { return b.Fault_ }

func Set(ctx context.Context, r soap.RoundTripper, req *types.Set) (*types.SetResponse, error) {
	var reqBody, resBody SetBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetLocaleBody struct {
	Req    *types.SetLocale         `xml:"urn:lookup SetLocale,omitempty"`
	Res    *types.SetLocaleResponse `xml:"urn:lookup SetLocaleResponse,omitempty"`
	Fault_ *soap.Fault              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetLocaleBody) Fault() *soap.Fault { return b.Fault_ }

func SetLocale(ctx context.Context, r soap.RoundTripper, req *types.SetLocale) (*types.SetLocaleResponse, error) {
	var reqBody, resBody SetLocaleBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}
