/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

	"github.com/vmware/govmomi/ssoadmin/types"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/xml"
)

// Methods here are not included in the wsdl

type GrantWSTrustRoleBody struct {
	Req    *types.GrantWSTrustRole         `xml:"urn:sso GrantWSTrustRole,omitempty"`
	Res    *types.GrantWSTrustRoleResponse `xml:"urn:sso GrantWSTrustRoleResponse,omitempty"`
	Fault_ *soap.Fault                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GrantWSTrustRoleBody) Fault() *soap.Fault { return b.Fault_ }

func GrantWSTrustRole(ctx context.Context, r soap.RoundTripper, req *types.GrantWSTrustRole) (*types.GrantWSTrustRoleResponse, error) {
	var reqBody, resBody GrantWSTrustRoleBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type RevokeWSTrustRoleBody struct {
	Req    *types.RevokeWSTrustRole         `xml:"urn:sso RevokeWSTrustRole,omitempty"`
	Res    *types.RevokeWSTrustRoleResponse `xml:"urn:sso RevokeWSTrustRoleResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *RevokeWSTrustRoleBody) Fault() *soap.Fault { return b.Fault_ }

func RevokeWSTrustRole(ctx context.Context, r soap.RoundTripper, req *types.RevokeWSTrustRole) (*types.RevokeWSTrustRoleResponse, error) {
	var reqBody, resBody RevokeWSTrustRoleBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

// C14N returns the canonicalized form of LoginBody.Req, for use by sts.Signer
func (b *LoginBody) C14N() string {
	req, err := xml.Marshal(b.Req)
	if err != nil {
		panic(err)
	}
	return string(req)
}
