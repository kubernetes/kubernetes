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

package disk

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/units"
	"github.com/vmware/govmomi/vim25/types"
)

type create struct {
	*flags.DatastoreFlag

	Bytes units.ByteSize
}

func init() {
	cli.Register("datastore.disk.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	_ = cmd.Bytes.Set("10G")
	f.Var(&cmd.Bytes, "size", "Size of new disk")
}

func (cmd *create) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *create) Usage() string {
	return "VMDK"
}

func (cmd *create) Description() string {
	return `Create VMDK on DS.

Examples:
  govc datastore.mkdir disks
  govc datastore.disk.create -size 24G disks/disk1.vmdk`
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 {
		return flag.ErrHelp
	}

	dc, err := cmd.Datacenter()
	if err != nil {
		return err
	}

	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	m := object.NewVirtualDiskManager(ds.Client())

	spec := &types.FileBackedVirtualDiskSpec{
		VirtualDiskSpec: types.VirtualDiskSpec{
			AdapterType: string(types.VirtualDiskAdapterTypeLsiLogic),
			DiskType:    string(types.VirtualDiskTypeThin),
		},
		CapacityKb: int64(cmd.Bytes) / 1024,
	}

	task, err := m.CreateVirtualDisk(ctx, ds.Path(f.Arg(0)), dc, spec)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}
