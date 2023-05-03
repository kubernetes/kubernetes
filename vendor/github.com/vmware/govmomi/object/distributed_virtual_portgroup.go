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
	"context"
	"fmt"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type DistributedVirtualPortgroup struct {
	Common
}

func NewDistributedVirtualPortgroup(c *vim25.Client, ref types.ManagedObjectReference) *DistributedVirtualPortgroup {
	return &DistributedVirtualPortgroup{
		Common: NewCommon(c, ref),
	}
}

func (p DistributedVirtualPortgroup) GetInventoryPath() string {
	return p.InventoryPath
}

// EthernetCardBackingInfo returns the VirtualDeviceBackingInfo for this DistributedVirtualPortgroup
func (p DistributedVirtualPortgroup) EthernetCardBackingInfo(ctx context.Context) (types.BaseVirtualDeviceBackingInfo, error) {
	var dvp mo.DistributedVirtualPortgroup
	var dvs mo.DistributedVirtualSwitch
	prop := "config.distributedVirtualSwitch"

	if err := p.Properties(ctx, p.Reference(), []string{"key", prop}, &dvp); err != nil {
		return nil, err
	}

	// From the docs at https://code.vmware.com/apis/196/vsphere/doc/vim.dvs.DistributedVirtualPortgroup.ConfigInfo.html:
	// "This property should always be set unless the user's setting does not have System.Read privilege on the object referred to by this property."
	// Note that "the object" refers to the Switch, not the PortGroup.
	if dvp.Config.DistributedVirtualSwitch == nil {
		name := p.InventoryPath
		if name == "" {
			name = p.Reference().String()
		}
		return nil, fmt.Errorf("failed to create EthernetCardBackingInfo for %s: System.Read privilege required for %s", name, prop)
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

func (p DistributedVirtualPortgroup) Reconfigure(ctx context.Context, spec types.DVPortgroupConfigSpec) (*Task, error) {
	req := types.ReconfigureDVPortgroup_Task{
		This: p.Reference(),
		Spec: spec,
	}

	res, err := methods.ReconfigureDVPortgroup_Task(ctx, p.Client(), &req)
	if err != nil {
		return nil, err
	}

	return NewTask(p.Client(), res.Returnval), nil
}
