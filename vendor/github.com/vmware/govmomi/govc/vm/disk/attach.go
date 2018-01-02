/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package disk

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type attach struct {
	*flags.DatastoreFlag
	*flags.VirtualMachineFlag

	persist    bool
	link       bool
	disk       string
	controller string
}

func init() {
	cli.Register("vm.disk.attach", &attach{})
}

func (cmd *attach) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.BoolVar(&cmd.persist, "persist", true, "Persist attached disk")
	f.BoolVar(&cmd.link, "link", true, "Link specified disk")
	f.StringVar(&cmd.controller, "controller", "", "Disk controller")
	f.StringVar(&cmd.disk, "disk", "", "Disk path name")
}

func (cmd *attach) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *attach) Run(ctx context.Context, f *flag.FlagSet) error {
	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return flag.ErrHelp
	}

	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	devices, err := vm.Device(ctx)
	if err != nil {
		return err
	}

	controller, err := devices.FindDiskController(cmd.controller)
	if err != nil {
		return err
	}

	disk := devices.CreateDisk(controller, ds.Reference(), ds.Path(cmd.disk))
	backing := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo)

	if cmd.link {
		if cmd.persist {
			backing.DiskMode = string(types.VirtualDiskModeIndependent_persistent)
		} else {
			backing.DiskMode = string(types.VirtualDiskModeIndependent_nonpersistent)
		}

		disk = devices.ChildDisk(disk)
		return vm.AddDevice(ctx, disk)
	}

	if cmd.persist {
		backing.DiskMode = string(types.VirtualDiskModePersistent)
	} else {
		backing.DiskMode = string(types.VirtualDiskModeNonpersistent)
	}

	return vm.AddDevice(ctx, disk)
}
