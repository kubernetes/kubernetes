/*
Copyright (c) 2019 VMware, Inc. All Rights Reserved.

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
	"fmt"
	"log"
	"math"
	"strconv"
	"strings"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/ovf"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type OvfManager struct {
	mo.OvfManager
}

func ovfDisk(e *ovf.Envelope, diskID string) *ovf.VirtualDiskDesc {
	for _, disk := range e.Disk.Disks {
		if strings.HasSuffix(diskID, disk.DiskID) {
			return &disk
		}
	}
	return nil
}

func ovfNetwork(ctx *Context, req *types.CreateImportSpec, item ovf.ResourceAllocationSettingData) types.BaseVirtualDeviceBackingInfo {
	if len(item.Connection) == 0 {
		return nil
	}
	pool := ctx.Map.Get(req.ResourcePool).(mo.Entity)
	ref := ctx.Map.getEntityDatacenter(pool).defaultNetwork()[0] // Default to VM Network
	c := item.Connection[0]

	for _, net := range req.Cisp.NetworkMapping {
		if net.Name == c {
			ref = net.Network
			break
		}
	}

	switch obj := ctx.Map.Get(ref).(type) {
	case *mo.Network:
		return &types.VirtualEthernetCardNetworkBackingInfo{
			VirtualDeviceDeviceBackingInfo: types.VirtualDeviceDeviceBackingInfo{
				DeviceName: obj.Name,
			},
		}
	case *DistributedVirtualPortgroup:
		dvs := ctx.Map.Get(*obj.Config.DistributedVirtualSwitch).(*DistributedVirtualSwitch)
		return &types.VirtualEthernetCardDistributedVirtualPortBackingInfo{
			Port: types.DistributedVirtualSwitchPortConnection{
				PortgroupKey: obj.Key,
				SwitchUuid:   dvs.Config.GetDVSConfigInfo().Uuid,
			},
		}
	default:
		log.Printf("ovf: unknown network type: %T", ref)
		return nil
	}
}

func ovfDiskCapacity(disk *ovf.VirtualDiskDesc) int64 {
	b, _ := strconv.ParseUint(disk.Capacity, 10, 64)
	if disk.CapacityAllocationUnits == nil {
		return int64(b)
	}
	c := strings.Fields(*disk.CapacityAllocationUnits)
	if len(c) == 3 && c[0] == "byte" && c[1] == "*" { // "byte * 2^20"
		p := strings.Split(c[2], "^")
		x, _ := strconv.ParseUint(p[0], 10, 64)
		if len(p) == 2 {
			y, _ := strconv.ParseUint(p[1], 10, 64)
			b *= uint64(math.Pow(float64(x), float64(y)))
		} else {
			b *= x
		}
	}
	return int64(b / 1024)
}

func (m *OvfManager) CreateImportSpec(ctx *Context, req *types.CreateImportSpec) soap.HasFault {
	body := new(methods.CreateImportSpecBody)

	env, err := ovf.Unmarshal(strings.NewReader(req.OvfDescriptor))
	if err != nil {
		body.Fault_ = Fault(err.Error(), &types.InvalidArgument{InvalidProperty: "ovfDescriptor"})
		return body
	}

	ds := ctx.Map.Get(req.Datastore).(*Datastore)
	path := object.DatastorePath{Datastore: ds.Name}
	spec := &types.VirtualMachineImportSpec{
		ConfigSpec: types.VirtualMachineConfigSpec{
			Name:    req.Cisp.EntityName,
			Version: esx.HardwareVersion,
			GuestId: string(types.VirtualMachineGuestOsIdentifierOtherGuest),
			Files: &types.VirtualMachineFileInfo{
				VmPathName: path.String(),
			},
			NumCPUs:           1,
			NumCoresPerSocket: 1,
			MemoryMB:          32,
		},
		ResPoolEntity: &req.ResourcePool,
	}

	if req.Cisp.DeploymentOption == "" && env.DeploymentOption != nil {
		for _, c := range env.DeploymentOption.Configuration {
			if isTrue(c.Default) {
				req.Cisp.DeploymentOption = c.ID
				break
			}
		}
	}

	if os := env.VirtualSystem.OperatingSystem; len(os) != 0 {
		if id := os[0].OSType; id != nil {
			spec.ConfigSpec.GuestId = *id
		}
	}

	var device object.VirtualDeviceList
	result := types.OvfCreateImportSpecResult{
		ImportSpec: spec,
	}

	hw := env.VirtualSystem.VirtualHardware[0]
	if vmx := hw.System.VirtualSystemType; vmx != nil {
		spec.ConfigSpec.Version = *vmx
	}

	ndisk := 0
	ndev := 0
	resources := make(map[string]types.BaseVirtualDevice)

	for _, item := range hw.Item {
		if req.Cisp.DeploymentOption != "" && item.Configuration != nil {
			if req.Cisp.DeploymentOption != *item.Configuration {
				continue
			}
		}

		kind := func() string {
			if item.ResourceSubType == nil {
				return "unknown"
			}
			return strings.ToLower(*item.ResourceSubType)
		}

		unsupported := func(err error) {
			result.Error = append(result.Error, types.LocalizedMethodFault{
				Fault: &types.OvfUnsupportedType{
					Name:       item.ElementName,
					InstanceId: item.InstanceID,
					DeviceType: int32(*item.ResourceType),
				},
				LocalizedMessage: err.Error(),
			})
		}

		upload := func(file ovf.File, c types.BaseVirtualDevice, n int) {
			result.FileItem = append(result.FileItem, types.OvfFileItem{
				DeviceId: fmt.Sprintf("/%s/%s:%d", req.Cisp.EntityName, device.Type(c), n),
				Path:     file.Href,
				Size:     int64(file.Size),
				CimType:  int32(*item.ResourceType),
			})
		}

		switch *item.ResourceType {
		case 1: // VMCI
		case 3: // Number of Virtual CPUs
			spec.ConfigSpec.NumCPUs = int32(*item.VirtualQuantity)
		case 4: // Memory Size
			spec.ConfigSpec.MemoryMB = int64(*item.VirtualQuantity)
		case 5: // IDE Controller
			d, _ := device.CreateIDEController()
			device = append(device, d)
			resources[item.InstanceID] = d
		case 6: // SCSI Controller
			d, err := device.CreateSCSIController(kind())
			if err == nil {
				device = append(device, d)
				resources[item.InstanceID] = d
			} else {
				unsupported(err)
			}
		case 10: // Virtual Network
			net := ovfNetwork(ctx, req, item)
			if net != nil {
				d, err := device.CreateEthernetCard(kind(), net)
				if err == nil {
					device = append(device, d)
				} else {
					unsupported(err)
				}
			}
		case 14: // Floppy Drive
			if device.PickController((*types.VirtualSIOController)(nil)) == nil {
				c := &types.VirtualSIOController{}
				c.Key = device.NewKey()
				device = append(device, c)
			}
			d, err := device.CreateFloppy()
			if err == nil {
				device = append(device, d)
				resources[item.InstanceID] = d
			} else {
				unsupported(err)
			}
		case 15: // CD/DVD
			c, ok := resources[*item.Parent]
			if !ok {
				continue // Parent is unsupported()
			}
			d, _ := device.CreateCdrom(c.(*types.VirtualIDEController))
			if len(item.HostResource) != 0 {
				for _, file := range env.References {
					if strings.HasSuffix(item.HostResource[0], file.ID) {
						path.Path = fmt.Sprintf("%s/_deviceImage%d.iso", req.Cisp.EntityName, ndev)
						device.InsertIso(d, path.String())
						upload(file, d, ndev)
						break
					}
				}
			}
			device = append(device, d)
			ndev++
		case 17: // Virtual Disk
			c, ok := resources[*item.Parent]
			if !ok {
				continue // Parent is unsupported()
			}
			path.Path = fmt.Sprintf("%s/disk-%d.vmdk", req.Cisp.EntityName, ndisk)
			d := device.CreateDisk(c.(types.BaseVirtualController), ds.Reference(), path.String())
			d.VirtualDevice.DeviceInfo = &types.Description{
				Label: item.ElementName,
			}
			disk := ovfDisk(env, item.HostResource[0])
			for _, file := range env.References {
				if file.ID == *disk.FileRef {
					upload(file, d, ndisk)
					break
				}
			}
			d.CapacityInKB = ovfDiskCapacity(disk)
			device = append(device, d)
			ndisk++
		case 23: // USB Controller
		case 24: // Video Card
		default:
			unsupported(fmt.Errorf("unsupported resource type: %d", *item.ResourceType))
		}
	}

	spec.ConfigSpec.DeviceChange, _ = device.ConfigSpec(types.VirtualDeviceConfigSpecOperationAdd)

	for _, p := range req.Cisp.PropertyMapping {
		spec.ConfigSpec.ExtraConfig = append(spec.ConfigSpec.ExtraConfig, &types.OptionValue{
			Key:   p.Key,
			Value: p.Value,
		})
	}

	body.Res = &types.CreateImportSpecResponse{
		Returnval: result,
	}

	return body
}
