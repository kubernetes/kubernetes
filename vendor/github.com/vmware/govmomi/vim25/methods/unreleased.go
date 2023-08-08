/*
   Copyright (c) 2022 VMware, Inc. All Rights Reserved.

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

type PlaceVmsXClusterBody struct {
	Req    *types.PlaceVmsXCluster         `xml:"urn:vim25 PlaceVmsXCluster,omitempty"`
	Res    *types.PlaceVmsXClusterResponse `xml:"PlaceVmsXClusterResponse,omitempty"`
	Fault_ *soap.Fault                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PlaceVmsXClusterBody) Fault() *soap.Fault { return b.Fault_ }

func PlaceVmsXCluster(ctx context.Context, r soap.RoundTripper, req *types.PlaceVmsXCluster) (*types.PlaceVmsXClusterResponse, error) {
	var reqBody, resBody PlaceVmsXClusterBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}
