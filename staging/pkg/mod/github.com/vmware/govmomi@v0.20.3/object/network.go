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

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type Network struct {
	Common
}

func NewNetwork(c *vim25.Client, ref types.ManagedObjectReference) *Network {
	return &Network{
		Common: NewCommon(c, ref),
	}
}

// EthernetCardBackingInfo returns the VirtualDeviceBackingInfo for this Network
func (n Network) EthernetCardBackingInfo(ctx context.Context) (types.BaseVirtualDeviceBackingInfo, error) {
	var e mo.Network

	// Use Network.Name rather than Common.Name as the latter does not return the complete name if it contains a '/'
	// We can't use Common.ObjectName here either as we need the ManagedEntity.Name field is not set since mo.Network
	// has its own Name field.
	err := n.Properties(ctx, n.Reference(), []string{"name"}, &e)
	if err != nil {
		return nil, err
	}

	backing := &types.VirtualEthernetCardNetworkBackingInfo{
		VirtualDeviceDeviceBackingInfo: types.VirtualDeviceDeviceBackingInfo{
			DeviceName: e.Name,
		},
	}

	return backing, nil
}
