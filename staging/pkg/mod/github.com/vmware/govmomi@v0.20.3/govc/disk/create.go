/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/units"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vslm"
)

type disk struct {
	*flags.DatastoreFlag
	*flags.ResourcePoolFlag
	*flags.StoragePodFlag

	size units.ByteSize
	keep *bool
}

func init() {
	cli.Register("disk.create", &disk{})
}

func (cmd *disk) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	cmd.StoragePodFlag, ctx = flags.NewStoragePodFlag(ctx)
	cmd.StoragePodFlag.Register(ctx, f)

	cmd.ResourcePoolFlag, ctx = flags.NewResourcePoolFlag(ctx)
	cmd.ResourcePoolFlag.Register(ctx, f)

	_ = cmd.size.Set("10G")
	f.Var(&cmd.size, "size", "Size of new disk")
	f.Var(flags.NewOptionalBool(&cmd.keep), "keep", "Keep disk after VM is deleted")
}

func (cmd *disk) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.StoragePodFlag.Process(ctx); err != nil {
		return err
	}
	return cmd.ResourcePoolFlag.Process(ctx)
}

func (cmd *disk) Usage() string {
	return "NAME"
}

func (cmd *disk) Description() string {
	return `Create disk NAME on DS.

Examples:
  govc disk.create -size 24G my-disk`
}

func (cmd *disk) Run(ctx context.Context, f *flag.FlagSet) error {
	name := f.Arg(0)
	if name == "" {
		return flag.ErrHelp
	}

	c, err := cmd.DatastoreFlag.Client()
	if err != nil {
		return err
	}

	var pool *object.ResourcePool
	var ds mo.Reference
	if cmd.StoragePodFlag.Isset() {
		ds, err = cmd.StoragePod()
		if err != nil {
			return err
		}
		pool, err = cmd.ResourcePool()
		if err != nil {
			return err
		}
	} else {
		ds, err = cmd.Datastore()
		if err != nil {
			return err
		}
	}

	m := vslm.NewObjectManager(c)

	spec := types.VslmCreateSpec{
		Name:              name,
		CapacityInMB:      int64(cmd.size) / units.MB,
		KeepAfterDeleteVm: cmd.keep,
		BackingSpec: &types.VslmCreateSpecDiskFileBackingSpec{
			VslmCreateSpecBackingSpec: types.VslmCreateSpecBackingSpec{
				Datastore: ds.Reference(),
			},
			ProvisioningType: string(types.BaseConfigInfoDiskFileBackingInfoProvisioningTypeThin),
		},
	}

	if cmd.StoragePodFlag.Isset() {
		if err = m.PlaceDisk(ctx, &spec, pool.Reference()); err != nil {
			return err
		}
	}

	task, err := m.CreateDisk(ctx, spec)
	if err != nil {
		return nil
	}

	logger := cmd.DatastoreFlag.ProgressLogger(fmt.Sprintf("Creating %s...", spec.Name))

	res, err := task.WaitForResult(ctx, logger)
	logger.Wait()
	if err != nil {
		return err
	}

	fmt.Println(res.Result.(types.VStorageObject).Config.Id.Id)

	return nil
}
