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
	"github.com/vmware/govmomi/vim25/types"
)

type Reference interface {
	Reference() types.ManagedObjectReference
}

func NewReference(c *vim25.Client, e types.ManagedObjectReference) Reference {
	switch e.Type {
	case "Folder":
		return NewFolder(c, e)
	case "StoragePod":
		return &StoragePod{
			NewFolder(c, e),
		}
	case "Datacenter":
		return NewDatacenter(c, e)
	case "VirtualMachine":
		return NewVirtualMachine(c, e)
	case "VirtualApp":
		return &VirtualApp{
			NewResourcePool(c, e),
		}
	case "ComputeResource":
		return NewComputeResource(c, e)
	case "ClusterComputeResource":
		return NewClusterComputeResource(c, e)
	case "HostSystem":
		return NewHostSystem(c, e)
	case "Network", "OpaqueNetwork":
		return NewNetwork(c, e)
	case "ResourcePool":
		return NewResourcePool(c, e)
	case "DistributedVirtualSwitch":
		return NewDistributedVirtualSwitch(c, e)
	case "VmwareDistributedVirtualSwitch":
		return &VmwareDistributedVirtualSwitch{*NewDistributedVirtualSwitch(c, e)}
	case "DistributedVirtualPortgroup":
		return NewDistributedVirtualPortgroup(c, e)
	case "Datastore":
		return NewDatastore(c, e)
	default:
		panic("Unknown managed entity: " + e.Type)
	}
}
