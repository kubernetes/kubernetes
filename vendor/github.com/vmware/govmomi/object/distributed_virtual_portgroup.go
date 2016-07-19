/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type DistributedVirtualPortgroup struct {
	Common
}

func NewDistributedVirtualPortgroup(c *vim25.Client, ref types.ManagedObjectReference) *DistributedVirtualPortgroup {
	return &DistributedVirtualPortgroup{
		Common: NewCommon(c, ref),
	}
}

// EthernetCardBackingInfo returns the VirtualDeviceBackingInfo for this DistributedVirtualPortgroup
func (p DistributedVirtualPortgroup) EthernetCardBackingInfo(ctx context.Context) (types.BaseVirtualDeviceBackingInfo, error) {
	var dvp mo.DistributedVirtualPortgroup
	var dvs mo.VmwareDistributedVirtualSwitch // TODO: should be mo.BaseDistributedVirtualSwitch

	if err := p.Properties(ctx, p.Reference(), []string{"key", "config.distributedVirtualSwitch"}, &dvp); err != nil {
		return nil, err
	}

	if err := p.Properties(ctx, *dvp.Config.DistributedVirtualSwitch, []string{"uuid"}, &dvs); err != nil {
		return nil, err
	}

	backing := &types.VirtualEthernetCardDistributedVirtualPortBackingInfo{
		Port: types.DistributedVirtualSwitchPortConnection{
			PortgroupKey: dvp.Key,
			SwitchUuid:   dvs.Uuid,
		},
	}

	return backing, nil
}
