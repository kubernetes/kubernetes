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
	"fmt"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type OpaqueNetwork struct {
	Common
}

func NewOpaqueNetwork(c *vim25.Client, ref types.ManagedObjectReference) *OpaqueNetwork {
	return &OpaqueNetwork{
		Common: NewCommon(c, ref),
	}
}

// EthernetCardBackingInfo returns the VirtualDeviceBackingInfo for this Network
func (n OpaqueNetwork) EthernetCardBackingInfo(ctx context.Context) (types.BaseVirtualDeviceBackingInfo, error) {
	var net mo.OpaqueNetwork

	if err := n.Properties(ctx, n.Reference(), []string{"summary"}, &net); err != nil {
		return nil, err
	}

	summary, ok := net.Summary.(*types.OpaqueNetworkSummary)
	if !ok {
		return nil, fmt.Errorf("%s unsupported network type: %T", n, net.Summary)
	}

	backing := &types.VirtualEthernetCardOpaqueNetworkBackingInfo{
		OpaqueNetworkId:   summary.OpaqueNetworkId,
		OpaqueNetworkType: summary.OpaqueNetworkType,
	}

	return backing, nil
}
