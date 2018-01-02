/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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

package vm

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/units"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type create struct {
	*flags.ClientFlag
	*flags.DatacenterFlag
	*flags.DatastoreFlag
	*flags.StoragePodFlag
	*flags.ResourcePoolFlag
	*flags.HostSystemFlag
	*flags.NetworkFlag
	*flags.FolderFlag

	name       string
	memory     int
	cpus       int
	guestID    string
	link       bool
	on         bool
	force      bool
	controller string
	annotation string

	iso              string
	isoDatastoreFlag *flags.DatastoreFlag
	isoDatastore     *object.Datastore

	disk              string
	diskDatastoreFlag *flags.DatastoreFlag
	diskDatastore     *object.Datastore

	// Only set if the disk argument is a byte size, which means the disk
	// doesn't exist yet and should be created
	diskByteSize int64

	Client       *vim25.Client
	Datacenter   *object.Datacenter
	Datastore    *object.Datastore
	StoragePod   *object.StoragePod
	ResourcePool *object.ResourcePool
	HostSystem   *object.HostSystem
	Folder       *object.Folder
}

func init() {
	cli.Register("vm.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	cmd.StoragePodFlag, ctx = flags.NewStoragePodFlag(ctx)
	cmd.StoragePodFlag.Register(ctx, f)

	cmd.ResourcePoolFlag, ctx = flags.NewResourcePoolFlag(ctx)
	cmd.ResourcePoolFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	cmd.NetworkFlag, ctx = flags.NewNetworkFlag(ctx)
	cmd.NetworkFlag.Register(ctx, f)

	cmd.FolderFlag, ctx = flags.NewFolderFlag(ctx)
	cmd.FolderFlag.Register(ctx, f)

	f.IntVar(&cmd.memory, "m", 1024, "Size in MB of memory")
	f.IntVar(&cmd.cpus, "c", 1, "Number of CPUs")
	f.StringVar(&cmd.guestID, "g", "otherGuest", "Guest OS")
	f.BoolVar(&cmd.link, "link", true, "Link specified disk")
	f.BoolVar(&cmd.on, "on", true, "Power on VM. Default is true if -disk argument is given.")
	f.BoolVar(&cmd.force, "force", false, "Create VM if vmx already exists")
	f.StringVar(&cmd.controller, "disk.controller", "scsi", "Disk controller type")
	f.StringVar(&cmd.annotation, "annotation", "", "VM description")

	f.StringVar(&cmd.iso, "iso", "", "ISO path")
	cmd.isoDatastoreFlag, ctx = flags.NewCustomDatastoreFlag(ctx)
	f.StringVar(&cmd.isoDatastoreFlag.Name, "iso-datastore", "", "Datastore for ISO file")

	f.StringVar(&cmd.disk, "disk", "", "Disk path (to use existing) OR size (to create new, e.g. 20GB)")
	cmd.diskDatastoreFlag, _ = flags.NewCustomDatastoreFlag(ctx)
	f.StringVar(&cmd.diskDatastoreFlag.Name, "disk-datastore", "", "Datastore for disk file")
}

func (cmd *create) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.StoragePodFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ResourcePoolFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.NetworkFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.FolderFlag.Process(ctx); err != nil {
		return err
	}

	// Default iso/disk datastores to the VM's datastore
	if cmd.isoDatastoreFlag.Name == "" {
		cmd.isoDatastoreFlag = cmd.DatastoreFlag
	}
	if cmd.diskDatastoreFlag.Name == "" {
		cmd.diskDatastoreFlag = cmd.DatastoreFlag
	}

	return nil
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	var err error

	if len(f.Args()) != 1 {
		return flag.ErrHelp
	}

	cmd.name = f.Arg(0)
	if cmd.name == "" {
		return flag.ErrHelp
	}

	cmd.Client, err = cmd.ClientFlag.Client()
	if err != nil {
		return err
	}

	cmd.Datacenter, err = cmd.DatacenterFlag.Datacenter()
	if err != nil {
		return err
	}

	if cmd.StoragePodFlag.Isset() {
		cmd.StoragePod, err = cmd.StoragePodFlag.StoragePod()
		if err != nil {
			return err
		}
	} else {
		cmd.Datastore, err = cmd.DatastoreFlag.Datastore()
		if err != nil {
			return err
		}
	}

	cmd.HostSystem, err = cmd.HostSystemFlag.HostSystemIfSpecified()
	if err != nil {
		return err
	}

	if cmd.HostSystem != nil {
		if cmd.ResourcePool, err = cmd.HostSystem.ResourcePool(ctx); err != nil {
			return err
		}
	} else {
		// -host is optional
		if cmd.ResourcePool, err = cmd.ResourcePoolFlag.ResourcePool(); err != nil {
			return err
		}
	}

	if cmd.Folder, err = cmd.FolderFlag.Folder(); err != nil {
		return err
	}

	// Verify ISO exists
	if cmd.iso != "" {
		_, err = cmd.isoDatastoreFlag.Stat(ctx, cmd.iso)
		if err != nil {
			return err
		}

		cmd.isoDatastore, err = cmd.isoDatastoreFlag.Datastore()
		if err != nil {
			return err
		}
	}

	// Verify disk exists
	if cmd.disk != "" {
		var b units.ByteSize

		// If disk can be parsed as byte units, don't stat
		err = b.Set(cmd.disk)
		if err == nil {
			cmd.diskByteSize = int64(b)
		} else {
			_, err = cmd.diskDatastoreFlag.Stat(ctx, cmd.disk)
			if err != nil {
				return err
			}

			cmd.diskDatastore, err = cmd.diskDatastoreFlag.Datastore()
			if err != nil {
				return err
			}
		}
	}

	task, err := cmd.createVM(ctx)
	if err != nil {
		return err
	}

	info, err := task.WaitForResult(ctx, nil)
	if err != nil {
		return err
	}

	vm := object.NewVirtualMachine(cmd.Client, info.Result.(types.ManagedObjectReference))

	if cmd.on {
		task, err := vm.PowerOn(ctx)
		if err != nil {
			return err
		}

		_, err = task.WaitForResult(ctx, nil)
		if err != nil {
			return err
		}
	}

	return nil
}

func (cmd *create) createVM(ctx context.Context) (*object.Task, error) {
	var devices object.VirtualDeviceList
	var err error

	spec := &types.VirtualMachineConfigSpec{
		Name:       cmd.name,
		GuestId:    cmd.guestID,
		NumCPUs:    int32(cmd.cpus),
		MemoryMB:   int64(cmd.memory),
		Annotation: cmd.annotation,
	}

	devices, err = cmd.addStorage(nil)
	if err != nil {
		return nil, err
	}

	devices, err = cmd.addNetwork(devices)
	if err != nil {
		return nil, err
	}

	deviceChange, err := devices.ConfigSpec(types.VirtualDeviceConfigSpecOperationAdd)
	if err != nil {
		return nil, err
	}

	spec.DeviceChange = deviceChange

	var datastore *object.Datastore

	// If storage pod is specified, collect placement recommendations
	if cmd.StoragePod != nil {
		datastore, err = cmd.recommendDatastore(ctx, spec)
		if err != nil {
			return nil, err
		}
	} else {
		datastore = cmd.Datastore
	}

	if !cmd.force {
		vmxPath := fmt.Sprintf("%s/%s.vmx", cmd.name, cmd.name)

		_, err := datastore.Stat(ctx, vmxPath)
		if err == nil {
			dsPath := cmd.Datastore.Path(vmxPath)
			return nil, fmt.Errorf("File %s already exists", dsPath)
		}
	}

	folder := cmd.Folder

	spec.Files = &types.VirtualMachineFileInfo{
		VmPathName: fmt.Sprintf("[%s]", datastore.Name()),
	}

	return folder.CreateVM(ctx, *spec, cmd.ResourcePool, cmd.HostSystem)
}

func (cmd *create) addStorage(devices object.VirtualDeviceList) (object.VirtualDeviceList, error) {
	if cmd.controller != "ide" {
		if cmd.controller == "nvme" {
			nvme, err := devices.CreateNVMEController()
			if err != nil {
				return nil, err
			}

			devices = append(devices, nvme)
			cmd.controller = devices.Name(nvme)
		} else {
			scsi, err := devices.CreateSCSIController(cmd.controller)
			if err != nil {
				return nil, err
			}

			devices = append(devices, scsi)
			cmd.controller = devices.Name(scsi)
		}
	}

	// If controller is specified to be IDE or if an ISO is specified, add IDE controller.
	if cmd.controller == "ide" || cmd.iso != "" {
		ide, err := devices.CreateIDEController()
		if err != nil {
			return nil, err
		}

		devices = append(devices, ide)
	}

	if cmd.diskByteSize != 0 {
		controller, err := devices.FindDiskController(cmd.controller)
		if err != nil {
			return nil, err
		}

		disk := &types.VirtualDisk{
			VirtualDevice: types.VirtualDevice{
				Key: devices.NewKey(),
				Backing: &types.VirtualDiskFlatVer2BackingInfo{
					DiskMode:        string(types.VirtualDiskModePersistent),
					ThinProvisioned: types.NewBool(true),
				},
			},
			CapacityInKB: cmd.diskByteSize / 1024,
		}

		devices.AssignController(disk, controller)
		devices = append(devices, disk)
	} else if cmd.disk != "" {
		controller, err := devices.FindDiskController(cmd.controller)
		if err != nil {
			return nil, err
		}

		ds := cmd.diskDatastore.Reference()
		path := cmd.diskDatastore.Path(cmd.disk)
		disk := devices.CreateDisk(controller, ds, path)

		if cmd.link {
			disk = devices.ChildDisk(disk)
		}

		devices = append(devices, disk)
	}

	if cmd.iso != "" {
		ide, err := devices.FindIDEController("")
		if err != nil {
			return nil, err
		}

		cdrom, err := devices.CreateCdrom(ide)
		if err != nil {
			return nil, err
		}

		cdrom = devices.InsertIso(cdrom, cmd.isoDatastore.Path(cmd.iso))
		devices = append(devices, cdrom)
	}

	return devices, nil
}

func (cmd *create) addNetwork(devices object.VirtualDeviceList) (object.VirtualDeviceList, error) {
	netdev, err := cmd.NetworkFlag.Device()
	if err != nil {
		return nil, err
	}

	devices = append(devices, netdev)
	return devices, nil
}

func (cmd *create) recommendDatastore(ctx context.Context, spec *types.VirtualMachineConfigSpec) (*object.Datastore, error) {
	sp := cmd.StoragePod.Reference()

	// Build pod selection spec from config spec
	podSelectionSpec := types.StorageDrsPodSelectionSpec{
		StoragePod: &sp,
	}

	// Keep list of disks that need to be placed
	var disks []*types.VirtualDisk

	// Collect disks eligible for placement
	for _, deviceConfigSpec := range spec.DeviceChange {
		s := deviceConfigSpec.GetVirtualDeviceConfigSpec()
		if s.Operation != types.VirtualDeviceConfigSpecOperationAdd {
			continue
		}

		if s.FileOperation != types.VirtualDeviceConfigSpecFileOperationCreate {
			continue
		}

		d, ok := s.Device.(*types.VirtualDisk)
		if !ok {
			continue
		}

		podConfigForPlacement := types.VmPodConfigForPlacement{
			StoragePod: sp,
			Disk: []types.PodDiskLocator{
				{
					DiskId:          d.Key,
					DiskBackingInfo: d.Backing,
				},
			},
		}

		podSelectionSpec.InitialVmConfig = append(podSelectionSpec.InitialVmConfig, podConfigForPlacement)
		disks = append(disks, d)
	}

	sps := types.StoragePlacementSpec{
		Type:             string(types.StoragePlacementSpecPlacementTypeCreate),
		ResourcePool:     types.NewReference(cmd.ResourcePool.Reference()),
		PodSelectionSpec: podSelectionSpec,
		ConfigSpec:       spec,
	}

	srm := object.NewStorageResourceManager(cmd.Client)
	result, err := srm.RecommendDatastores(ctx, sps)
	if err != nil {
		return nil, err
	}

	// Use result to pin disks to recommended datastores
	recs := result.Recommendations
	if len(recs) == 0 {
		return nil, fmt.Errorf("no recommendations")
	}

	ds := recs[0].Action[0].(*types.StoragePlacementAction).Destination

	var mds mo.Datastore
	err = property.DefaultCollector(cmd.Client).RetrieveOne(ctx, ds, []string{"name"}, &mds)
	if err != nil {
		return nil, err
	}

	datastore := object.NewDatastore(cmd.Client, ds)
	datastore.InventoryPath = mds.Name

	// Apply recommendation to eligible disks
	for _, disk := range disks {
		backing := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo)
		backing.Datastore = &ds
	}

	return datastore, nil
}
