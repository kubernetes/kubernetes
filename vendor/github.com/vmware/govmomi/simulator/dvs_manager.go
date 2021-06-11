/*
   Copyright (c) 2020 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type DistributedVirtualSwitchManager struct {
	mo.DistributedVirtualSwitchManager
}

func (m *DistributedVirtualSwitchManager) DVSManagerLookupDvPortGroup(ctx *Context, req *types.DVSManagerLookupDvPortGroup) soap.HasFault {
	body := &methods.DVSManagerLookupDvPortGroupBody{}

	for _, obj := range ctx.Map.All("DistributedVirtualSwitch") {
		dvs := obj.(*DistributedVirtualSwitch)
		if dvs.Uuid == req.SwitchUuid {
			for _, ref := range dvs.Portgroup {
				pg := ctx.Map.Get(ref).(*DistributedVirtualPortgroup)
				if pg.Key == req.PortgroupKey {
					body.Res = &types.DVSManagerLookupDvPortGroupResponse{
						Returnval: &ref,
					}
					return body
				}
			}
		}
	}

	body.Fault_ = Fault("", new(types.NotFound))

	return body
}
